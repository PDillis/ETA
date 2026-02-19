import csv
import os
import re
import sys
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Union

# Silence noisy warnings before any imports that trigger them
warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*AccumulateGrad node's stream.*")
warnings.filterwarnings("ignore", message=".*Tensor Cores.*set.*torch.set_float32_matmul_precision.*")
warnings.filterwarnings("ignore", message=".*Checkpoint directory.*exists and is not empty.*")

# Add the models directory to sys.path so that 'cilpp' package can be imported
# This allows running the script from any directory
_SCRIPT_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _MODELS_DIR.parent  # For accessing agents/tools
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import click
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from lightning.pytorch import profilers

from rich.console import Console
from rich.panel import Panel

from cilpp.model import CILpp
from cilpp.data import CARLA_Data
from cilpp.config import GlobalConfig
from agents.tools.train_utils import (
    print_config_table,
    print_data_table,
    print_model_table,
    CSVLossLogger,
    make_run_dir,
    build_experiment_id,
    _worker_init_fn,
    _is_main_process,
    parse_comma_separated_ints,
)

# Avoid dataloader crash: Before creating loaders
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")  # avoids fd passing issues

console = Console()


class CILppPlanner(pl.LightningModule):
    """
    CIL++ trainer:
      - Model outputs (B, 1, len(config.targets))
      - Original L1 action loss on the last frame
      - AdamW + LR epoch decay
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cams = self.config.data_used
        self.model = CILpp(config)
        self.cmd_classes = self.config.data_command_class_num

        # Optimizer & LR policy
        self.lr = self.config.lr
        self.lr_min = self.config.lr_min
        self.lr_schedule = getattr(self.config, 'lr_schedule', 'none')
        self.lr_milestones = self.config.lr_milestones
        self.lr_decay_level = self.config.lr_decay_level
        self.num_epochs = getattr(self.config, 'num_epochs', 100)

        # Loss weights (original CIL++ used per-component weights)
        self.targets = list(self.config.targets)
        self.loss_weight = {
            "throttle": self.config.loss_weights["throttle"],
            "steer":    self.config.loss_weights["steer"],
            "brake":    self.config.loss_weights["brake"],
        }

        self.save_hyperparameters(ignore=["config"])  # keeps CLI args/logs tidy


    def forward(self, batch):
        pass

    def action_nospeed_l1(self, action_output: torch.Tensor, targets):
        """
        Predictions & targets are ordered [throttle, steer, brake].
        """
        B = action_output.size(0)

        # Take last frame
        pred = action_output[:, -1, :]     # [B, 3]

        # Grab weights
        w_thr = self.loss_weight["throttle"]
        w_ste = self.loss_weight["steer"]
        w_brk = self.loss_weight["brake"]

        # Compute per-dimension L1
        throttle_loss = F.l1_loss(pred[:, 0], targets[:, 0], reduction="mean") * w_thr
        steer_loss = F.l1_loss(pred[:, 1], targets[:, 1], reduction="mean") * w_ste
        brake_loss = F.l1_loss(pred[:, 2], targets[:, 2], reduction="mean") * w_brk

        loss = throttle_loss + steer_loss + brake_loss
        return loss, throttle_loss, steer_loss, brake_loss


    def _prepare_inputs(self, batch):
        """Prepare model inputs from a batch: imgs [B,S,Cam,C,H,W], command [B,num_cls], speed [B,1]."""
        # Stack camera views into [B, Cam, C, H, W], then add S=1 dim -> [B, 1, Cam, C, H, W]
        imgs = torch.stack([batch[cam] for cam in self.cams], dim=1).unsqueeze(1)
        command = self._ensure_one_hot(batch['target_command'], self.cmd_classes)
        speed = batch['speed'].to(dtype=torch.float32).view(-1, 1) / self.config.speed_max
        return imgs, command, speed

    def training_step(self, batch, batch_idx):
        imgs, command, speed = self._prepare_inputs(batch)

        # GT Actions: [B, 3] = [throttle, steer, brake]
        gt_actions = batch["action"].to(dtype=torch.float32)

        pred_act = self.model(imgs, command, speed)  # [B,1,3]

        loss, _, _, _ = self.action_nospeed_l1(pred_act, gt_actions)

        # Unweighted MAE per action (pure prediction error)
        pred_last = pred_act[:, -1, :]  # [B, 3]
        mae = torch.abs(pred_last - gt_actions)

        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/mae_throttle', mae[:, 0].mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/mae_steer', mae[:, 1].mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/mae_brake', mae[:, 2].mean(), on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)

        if self.lr_schedule == 'step':
            # torch.optim.lr_scheduler.MultiStepLR: decays LR by lr_decay_level at each milestone
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.lr_milestones,
                gamma=self.lr_decay_level,
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        if self.lr_schedule == 'cosine':
            # torch.optim.lr_scheduler.CosineAnnealingLR: cosine decay from lr to lr_min
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.num_epochs,
                eta_min=self.lr_min,
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        # lr_schedule == 'none': manual scheduling via on_train_epoch_start (backwards compatible)
        return optimizer

    def validation_step(self, batch, batch_idx):
        imgs, command, speed = self._prepare_inputs(batch)

        # GT Actions: [B, 3] = [throttle, steer, brake]
        gt_actions = batch["action"].to(dtype=torch.float32)

        pred_act, _, _ = self.model.forward_eval(imgs, command, speed)  # [B,1,3]

        loss, _, _, _ = self.action_nospeed_l1(pred_act, gt_actions)

        # Unweighted MAE per action (pure prediction error)
        pred_last = pred_act[:, -1, :]  # [B, 3]
        mae = torch.abs(pred_last - gt_actions)

        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/mae_throttle', mae[:, 0].mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/mae_steer', mae[:, 1].mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/mae_brake', mae[:, 2].mean(), on_step=False, on_epoch=True, sync_dist=True)

        return loss

    # CIL++ hand-made epoch-based LR decay (backwards compatible with previously trained models)
    def on_train_epoch_start(self):
        if self.lr_schedule != 'none':
            return
        if not self.lr_milestones:
            return
        # milestones are 1-based (common in epoch configs)
        if (self.current_epoch + 1) in set(int(e) for e in self.lr_milestones):
            opt = self.trainer.optimizers[0]
            for g in opt.param_groups:
                cur = float(g["lr"])
                new = max(cur * self.lr_decay_level, self.lr_min)
                g["lr"] = new
            for i, g in enumerate(opt.param_groups):
                self.log(f"lr_group{i}", g["lr"], on_epoch=True, sync_dist=True)

    def _ensure_one_hot(self, x: torch.Tensor, num_classes: int) -> torch.Tensor:
        if x.dim() == 2 and x.size(-1) == num_classes:
            return x.float()
        if x.dim() == 1 or (x.dim() == 2 and x.size(-1) == 1):
            ids = x.view(-1).long()
            return F.one_hot(ids, num_classes=num_classes).float()
        raise ValueError(f"Unexpected command tensor shape {tuple(x.shape)}")

    def _load_state_dict(self, il_net, rl_state_dict, key_word):
        rl_keys = [k for k in rl_state_dict.keys() if key_word in k]
        il_keys = il_net.state_dict().keys()
        assert len(rl_keys) == len(il_net.state_dict().keys()), f'mismatch number of layers loading {key_word}'
        new_state_dict = OrderedDict()
        for k_il, k_rl in zip(il_keys, rl_keys):
            new_state_dict[k_il] = rl_state_dict[k_rl]
        il_net.load_state_dict(new_state_dict)


@click.command()
# Required.
@click.option('--train-data',      help='Training data file (.npy)', metavar='PATH',            type=click.Path(exists=True), default=os.getenv('TRAIN_DATA_PATH'))
@click.option('--val-data',        help='Validation data file (.npy)', metavar='PATH',          type=click.Path(exists=True), default=os.getenv('VAL_DATA_PATH'))
# Optional features.
@click.option('--data-root',       help='Dataset root direct0ory (images)', metavar='DIR',      type=click.Path(exists=True), default=os.getenv('BENCH2DRIVE_ROOT'))
@click.option('--cameras',         help='Comma-separated camera names', metavar='STR',          type=str, default=None)
@click.option('--backbone',        help='Backbone architecture', metavar='STR',                 type=str, default=None)
@click.option('--img-size',        help='Image size as HxW (e.g., 300x300)', metavar='HxW',     type=str, default=None)
@click.option('--img-aug',         help='Enable image augmentation', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--tf32',            help='Enable TF32 matmul precision (Ampere+ GPUs)',          is_flag=True)
@click.option('--resume',          help='Resume interrupted training from run dir or .ckpt',    metavar='[DIR|CKPT]', type=str, default=None)
@click.option('--finetune',        help='Finetune from a pretrained checkpoint (fresh training)', metavar='CKPT', type=click.Path(exists=True), default=None)
# Hyperparameters.
@click.option('--batch-size',      help='Total batch size', metavar='INT',                      type=click.IntRange(min=1), default=128, show_default=True)
@click.option('--epochs',          help='Number of training epochs', metavar='INT',             type=click.IntRange(min=1), default=100, show_default=True)
@click.option('--lr',              help='Learning rate', metavar='FLOAT',                       type=click.FloatRange(min=0), default=1e-4, show_default=True)
@click.option('--lr-min',          help='Minimum learning rate', metavar='FLOAT',               type=click.FloatRange(min=0), default=1e-5, show_default=True)
@click.option('--lr-schedule',     help='LR schedule type', metavar='TYPE',                     type=click.Choice(['step', 'cosine', 'none']), default='step', show_default=True)
@click.option('--lr-milestones',   help='Epoch milestones for step decay (comma-sep)', metavar='STR', type=str, default='30,50,65', show_default=True)
@click.option('--lr-decay',        help='LR decay multiplier at each milestone', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0.5, show_default=True)
@click.option('--grad-clip',       help='Gradient clipping value (None=disabled)', metavar='FLOAT', type=float, default=None)
@click.option('--loss-w-throttle', help='Throttle loss weight', metavar='FLOAT',                type=click.FloatRange(min=0), default=0.25, show_default=True)
@click.option('--loss-w-steer',    help='Steer loss weight', metavar='FLOAT',                   type=click.FloatRange(min=0), default=0.50, show_default=True)
@click.option('--loss-w-brake',    help='Brake loss weight', metavar='FLOAT',                   type=click.FloatRange(min=0), default=0.25, show_default=True)
# Misc settings.
@click.option('--outdir',          help='Where to save the results', metavar='DIR',             type=click.Path(file_okay=False), default=os.getenv('TRAINING_LOG_DIR', os.path.join(os.getcwd(), 'training-runs')), show_default=True)
@click.option('--desc',            help='Extra string appended to run dir name', metavar='STR', type=str, default=None)
@click.option('--id', 'experiment_id', help='Base experiment identifier', metavar='STR',        type=str, default='CILpp', show_default=True)
@click.option('--gpus',            help='Number of GPUs', metavar='INT',                        type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--num-workers',     help='DataLoader worker processes', metavar='INT',           type=click.IntRange(min=0), default=16, show_default=True)
@click.option('--seed',            help='Random seed', metavar='INT',                           type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--val-every',       help='Validation frequency (epochs)', metavar='INT',         type=click.IntRange(min=1), default=5, show_default=True)
@click.option('--wandb-project',   help='Wandb project name', metavar='STR',                    type=str, default='CILpp', show_default=True)
@click.option('--no-wandb',        help='Disable wandb logging',                                is_flag=True)
@click.option('--profiler',        help='PyTorch Lightning profiler',                           type=click.Choice(['simple', 'advanced']), default=None)
@click.option('-n', '--dry-run',   help='Print training options and exit',                      is_flag=True)
def main(train_data: str, val_data: str, data_root: str,
         cameras: str, backbone: str, img_size: str, img_aug: bool, tf32: bool,
         resume: str, finetune: str,
         batch_size: int, epochs: int, lr: float, lr_min: float,
         lr_schedule: str, lr_milestones: str, lr_decay: float,
         grad_clip: float, loss_w_throttle: float, loss_w_steer: float, loss_w_brake: float,
         outdir: str, desc: str, experiment_id: str,
         gpus: int, num_workers: int, seed: int, val_every: int,
         wandb_project: str, no_wandb: bool, profiler: str, dry_run: bool):
    """Train CIL++ model for end-to-end autonomous driving."""

    # Validate required paths (must be provided via CLI or env var)
    if train_data is None:
        raise click.UsageError("--train-data is required (or set TRAIN_DATA_PATH env var)")
    if val_data is None:
        raise click.UsageError("--val-data is required (or set VAL_DATA_PATH env var)")

    # Reproducibility: seed Python, NumPy, PyTorch, and dataloader workers
    pl.seed_everything(seed, workers=True)

    # TF32 precision for Tensor Core GPUs (Ampere+)
    if tf32:
        # We need to go more fine-grained since torch >= 2.9
        torch.set_float32_matmul_precision('medium')
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        torch.backends.cudnn.rnn.fp32_precision = "tf32"

    # Config (CLI overrides config defaults)
    config = GlobalConfig()
    if data_root is not None:
        config.root_dir_all = data_root
    config.train_data = train_data
    config.val_data = val_data
    if cameras is not None:
        config.data_used = [c.strip() for c in cameras.split(',')]
    if img_size is not None:
        h, w = [int(x) for x in img_size.split('x')]
        config.image_shape = [3, h, w]
    if backbone is not None:
        config.backbone = backbone
    config.lr = lr
    config.lr_min = lr_min
    config.lr_schedule = lr_schedule
    config.lr_milestones = parse_comma_separated_ints(lr_milestones) if lr_schedule in ('step', 'none') else []
    config.lr_decay_level = lr_decay
    config.num_epochs = epochs
    config.img_aug = img_aug
    config.loss_weights = {"throttle": loss_w_throttle, "steer": loss_w_steer, "brake": loss_w_brake}

    # Build descriptive experiment ID
    experiment_id = build_experiment_id(experiment_id, config, batch_size, epochs, tf32, seed, img_aug)
    experiment_id = f"{experiment_id}_{desc}" if desc is not None else experiment_id

    # Resume vs Finetune vs New run
    # --resume: continue interrupted training (restores optimizer, epoch, etc.)
    # --finetune: load weights only, start fresh training with new config/epochs
    ckpt_path = None
    if resume is not None and finetune is not None:
        raise click.UsageError("Cannot use --resume and --finetune at the same time")

    # Determine rank early: DDP spawns multiple processes that all run main()
    is_main = _is_main_process()

    if resume is not None:
        if os.path.isdir(resume):
            ckpt_path = os.path.join(resume, 'last.ckpt')
            if not os.path.exists(ckpt_path):
                raise click.BadParameter(f"No last.ckpt found in {resume}", param_hint="'--resume'")
            run_dir = resume
        else:
            ckpt_path = resume
            run_dir = os.path.dirname(ckpt_path)
    else:
        # All ranks compute the run_dir path (they see the same filesystem),
        # but only rank 0 creates the directory to avoid DDP race conditions.
        run_dir = make_run_dir(outdir, experiment_id, dry_run=(not is_main or dry_run))

    # Extract run number for wandb name (e.g., "00003" from "00003-CILpp_...")
    run_name = os.path.basename(run_dir)
    if is_main:
        console.print(Panel.fit("[bold blue]CIL++[/bold blue] Training"))
        print_config_table(config, run_dir, batch_size, epochs, gpus, tf32=tf32,
                           lr_schedule=lr_schedule, grad_clip=grad_clip,
                           resume=resume, finetune=finetune)

    # Data
    if is_main:
        console.print("[dim]Loading datasets...[/dim]")
    train_set = CARLA_Data(
        root=config.root_dir_all,
        data_path=config.train_data,
        config=config,
        img_aug=config.img_aug,
        split="train",
        verbose=is_main,
    )
    val_set = CARLA_Data(
        root=config.root_dir_all,
        data_path=config.val_data,
        config=config,
        split="val",
        verbose=is_main,
    )

    if is_main:
        print_data_table(len(train_set), len(val_set), config.train_data, config.val_data)

    # Model
    if finetune is not None:
        model = CILppPlanner.load_from_checkpoint(finetune, config=config)
    else:
        model = CILppPlanner(config)
    if is_main:
        print_model_table(model)

    # Dry-run: print config and exit
    if dry_run:
        if is_main:
            console.print("\n[yellow]Dry run — exiting without training.[/yellow]")
        return

    use_persistent = num_workers > 0
    dataloader_train = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        sampler=None,  # TODO: test different sampling strategies (e.g., weighted, bins, etc.)
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=use_persistent,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn,
    )
    dataloader_val = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        save_weights_only=False,
        save_top_k=-1,
        every_n_epochs=val_every,
        save_last=True,
        dirpath=run_dir,
        filename=experiment_id + "_{epoch:03d}",
    )

    progress_bar = RichProgressBar(leave=True)

    csv_path = os.path.join(run_dir, "losses.csv")
    csv_logger = CSVLossLogger(csv_path)

    # Logger
    if no_wandb:
        logger = None
    else:
        logger = WandbLogger(
            project=wandb_project,
            name=run_name,
            save_dir=run_dir,
            save_code=False,
            config={
                "learning_rate": config.lr,
                "lr_min": config.lr_min,
                "lr_schedule": lr_schedule,
                "lr_milestones": config.lr_milestones,
                "lr_decay": lr_decay,
                "epochs": epochs,
                "batch_size": batch_size,
                "architecture": "CIL++",
                "backbone": config.backbone,
                "cameras": config.data_used,
                "image_shape": config.image_shape,
                "dataset": "Bench2Drive",
                "seq_len": config.seq_len,
                "pred_len": config.pred_len,
                "loss_weights": config.loss_weights,
                "grad_clip": grad_clip,
                "tf32": tf32,
                "seed": seed,
                "train_data": train_data,
                "val_data": val_data,
                "data_root": config.root_dir_all,
                "img_aug": config.img_aug,
                "finetune_from": finetune,
                "resumed_from": resume,
            },
        )
        _base = str(_MODELS_DIR.parent)
        logger.experiment.save(str(_SCRIPT_DIR / "config.py"), base_path=_base, policy="now")
        logger.experiment.save(str(_SCRIPT_DIR / "model.py"), base_path=_base, policy="now")
        logger.experiment.save(str(_SCRIPT_DIR / "train.py"), base_path=_base, policy="now")

    # Profiler
    pl_profiler = None
    if profiler == "simple":
        pl_profiler = profilers.SimpleProfiler(dirpath=run_dir, filename="simple_profiler")
    elif profiler == "advanced":
        pl_profiler = profilers.AdvancedProfiler(dirpath=run_dir, filename="advanced_profiler")

    # Trainer
    trainer = pl.Trainer(
        default_root_dir=run_dir,
        devices=gpus,
        accelerator='gpu',
        strategy=DDPStrategy(static_graph=True),
        sync_batchnorm=True,
        profiler=pl_profiler,
        benchmark=True,
        log_every_n_steps=1,
        logger=logger,
        callbacks=[checkpoint_callback, progress_bar, csv_logger],
        check_val_every_n_epoch=val_every,
        max_epochs=epochs,
        enable_progress_bar=True,
        enable_model_summary=False,
        gradient_clip_val=grad_clip,
        precision=16 if tf32 else 32,
    )

    if is_main:
        console.print()
        if resume is not None:
            console.rule(f"[bold green]Resuming Training from {os.path.basename(ckpt_path)}")
        elif finetune is not None:
            console.rule(f"[bold green]Finetuning from {os.path.basename(finetune)}")
        else:
            console.rule("[bold green]Starting Training")
        console.print()

    # Train the model
    trainer.fit(model, dataloader_train, dataloader_val, ckpt_path=ckpt_path)

    if is_main:
        console.print()
        console.rule("[bold green]Training Complete")
        console.print(f"[green]Checkpoints saved to:[/green] {run_dir}")
        console.print(f"[green]Loss history:[/green] {csv_path}")


if __name__ == "__main__":
    main()
