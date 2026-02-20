import torch
import torch.nn as nn
from einops import rearrange

from cilpp.building_blocks import FC
from cilpp.building_blocks.PositionalEncoding import PositionalEncoding
from cilpp.building_blocks.Transformer.TransformerEncoder import TransformerEncoder
from cilpp.building_blocks.Transformer.TransformerEncoder import TransformerEncoderLayer


RESNET_BACKBONES = {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}
VIT_BACKBONES = {"vit_tiny_16", "vit_small_16", "vit_base_16"}
DINO_BACKBONES = {"dinov3_small_16", "dinov3_smallplus_16", "dinov3_convnext_tiny", 
                  "dinov2_small_reg_14", "dinov2_base_reg_14",
                  "dino_small_8", "dino_small_16", "dino_base_16"}
THEIA_BACKBONES = {}


def create_backbone(backbone_name: str, pretrained: bool = True, layer_id: int = 4):
    """
    Factory for creating a vision backbone.

    All backbones must implement:
      - forward(x: [B,C,H,W]) -> (features: [B,D,h,w], intermediates: list)
      - get_backbone_output_shape(input_shape: [B,C,H,W]) -> list of shapes

    Args:
        backbone_name: e.g. 'resnet34', 'resnet50'. Future: 'vit_base_16', etc.
        pretrained: whether to load pretrained weights.
        layer_id: which layer to extract features from
        TODO: support multiple layer outputs (e.g., for FPN-style features) and different backbone types (e.g., ViT/Huggingface models).
    """
    if backbone_name in RESNET_BACKBONES:
        import cilpp.building_blocks.resnet as resnet_module
        constructor = getattr(resnet_module, backbone_name)
        return constructor(pretrained=pretrained, layer_id=layer_id)

    raise ValueError(
        f"Unknown backbone '{backbone_name}'. Supported: {sorted(RESNET_BACKBONES)}"
    )


class CILpp(nn.Module):
    """
    CIL++ (multiview) with Transformer encoder and an action MLP head.
    This mirrors the original CIL++ implementation semantics:
      - NO discrete action dictionary
      - Output: (B, 1, len(config.targets))
    """
	
    def __init__(self, config):
        super().__init__()
        self.params = config.model_configuration
        self.config = config 

        # ===== Perception backbone =====
        self.encoder_embedding_perception = create_backbone(
            backbone_name=self.config.backbone,
            pretrained=self.config.imagenet_pre_trained,
            layer_id=self.config.backbone_layer_id,
        )
        _, self.res_out_dim, self.res_out_h, self.res_out_w = \
            self.encoder_embedding_perception.get_backbone_output_shape(
                [self.config.batch_size] + self.config.image_shape
            )[self.config.backbone_layer_id]

        # Positional encoding
        self.d_model = self.params['TxEncoder']['d_model']
        L_tokens = (
            len(self.config.data_used)
            * self.config.seq_len
            * self.res_out_h
            * self.res_out_w
        )
        self.positional_encoding = self._create_positional_encoding(
            L_tokens, self.d_model, self.params['TxEncoder']['learnable_pe']
        )

        # CIL ++ measurements: Command/speed embeddings
        self.command = nn.Linear(self.config.data_command_class_num, self.d_model)
        self.speed = nn.Linear(1, self.d_model)

        # Transformer encoder
        tx_encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.params['TxEncoder']['n_head'],
            norm_first=self.params['TxEncoder']['norm_first'],
            batch_first=True
        )
        self.tx_encoder = TransformerEncoder(
            tx_encoder_layer,
            num_layers=self.params['TxEncoder']['num_layers'],
            norm=nn.LayerNorm(self.d_model)
        )

        # ===== Action head (FC) -> len(config.targets) =====
        join_dim = self.d_model
        self.action_output = FC(
            params={
                "neurons": [join_dim] 
                + self.params["action_output"]["fc"]["neurons"] 
                + [len(self.config.targets)],
                "dropouts": self.params["action_output"]["fc"]["dropouts"] + [0.0],
                "end_layer": True,
            }
        )

        # Mask prediction head (optional auxiliary task)
        self.mask_head = None
        if getattr(config, 'mask_loss_enabled', False):
            mask_size = config.mask_height * config.mask_width
            self.mask_head = nn.Sequential(
                nn.Linear(self.d_model, 256),
                nn.ReLU(),
                nn.Linear(256, mask_size),
            )
            self._mask_h = config.mask_height
            self._mask_w = config.mask_width

        # TODO: driving profile
        # TODO: add accelerometer data embedding
        # TODO: test additional tokens [STR], [THR], [BRK], [ACC], [ROT], etc.
        # TODO: ViT/Huggingface models

        # Init linear layers like CIL++
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    @staticmethod
    def _create_positional_encoding(num_tokens: int, d_model: int, learnable: bool):
        """
        Create positional encoding.

        Extension point for future 2D/3D PE variants. A 2D spatial PE would
        accept (h, w, num_cams, seq_len) and build a grid-based encoding.
        TODO: investigate 2D/3D PE variants (e.g., Rotary PE, etc.) and their impact on performance.
        """
        if learnable:
            return nn.Parameter(torch.zeros(1, num_tokens, d_model))
        return PositionalEncoding(d_model=d_model, dropout=0.0, max_len=num_tokens)

    def _encode(self, imgs, command, speed):
        """
        Shared encoder logic for train and eval.

        Args:
          imgs:    [B, S, Cam, C, H, W]  image sequence (S from config.seq_len)
          command: [B, data_command_class_num]  one-hot command (current frame)
          speed:   [B, 1]  normalized speed (current frame)

        Returns:
          enc:           [B, L, res_out_dim]  visual tokens
          e_d:           [B, 1, d_model]      command embedding
          e_s:           [B, 1, d_model]      speed embedding
          resnet_inter:  backbone intermediates
        """
        S = int(self.config.seq_len)

        # Flatten [B, S, Cam, C, H, W] -> [B*S*Cam, C, H, W] for backbone
        x = rearrange(imgs, 'B S Cam C H W -> (B S Cam) C H W')

        # Perception backbone
        e_p, resnet_inter = self.encoder_embedding_perception(x)  # [(B*S*Cam), res_out_dim, h, w]

        # Reshape to visual token sequence: [B, L, res_out_dim]
        # where L = S * Cam * h * w
        B = imgs.shape[0]
        Cams = len(self.config.data_used)
        enc = rearrange(
            e_p,
            '(B S Cam) D h w -> B (S Cam h w) D',
            B=B, S=S, Cam=Cams,
        )

        # Project command & speed to d_model (broadcast-added to all tokens)
        e_d = self.command(command).unsqueeze(1)  # [B, 1, d_model]
        e_s = self.speed(speed).unsqueeze(1)      # [B, 1, d_model]

        return enc, e_d, e_s, resnet_inter

    def forward(self, imgs, command, speed):
        """
        Args:
          imgs:    [B, S, Cam, C, H, W]  image sequence (S from config.seq_len)
          command: [B, data_command_class_num]  one-hot command (current frame)
          speed:   [B, 1]  normalized speed (current frame)

        Returns:
          action_output: [B, 1, len(config.targets)]
        """
        enc, e_d, e_s, _ = self._encode(imgs, command, speed)

        # CIL++ style: add measurement embeddings to all visual tokens
        tokens = enc + e_d + e_s  # [B, L, d_model]

        # Positional encoding
        if isinstance(self.positional_encoding, nn.Parameter):
            pe_tokens = tokens + self.positional_encoding
        else:
            pe_tokens = self.positional_encoding(tokens)

        # Transformer encoder
        tx_mem, _ = self.tx_encoder(pe_tokens)  # [B, L, d_model]

        # Global pooling over tokens
        in_memory = torch.mean(tx_mem, dim=1)  # [B, d_model]

        action_output = self.action_output(in_memory).unsqueeze(1)  # [B, 1, len(targets)]

        mask_logits = None
        if self.mask_head is not None:
            mask_logits = self.mask_head(in_memory).view(-1, self._mask_h, self._mask_w)

        return action_output, mask_logits

    def forward_eval(self, imgs, command, speed):
        """
        Evaluation forward that also returns intermediate backbone features and attention weights.

        Args:
          imgs:    [B, S, Cam, C, H, W]
          command: [B, data_command_class_num]
          speed:   [B, 1]

        Returns:
          action_output: [B, 1, len(config.targets)]
          resnet_inter:  backbone intermediates (as returned by encoder)
          attn_weights:  attention weights from TransformerEncoder
        """
        enc, e_d, e_s, resnet_inter = self._encode(imgs, command, speed)

        tokens = enc + e_d + e_s  # [B, L, d_model]

        if isinstance(self.positional_encoding, nn.Parameter):
            pe_tokens = tokens + self.positional_encoding
        else:
            pe_tokens = self.positional_encoding(tokens)

        tx_mem, attn_weights = self.tx_encoder(pe_tokens)  # [B, L, d_model]

        in_memory = torch.mean(tx_mem, dim=1)  # [B, d_model]

        action_output = self.action_output(in_memory).unsqueeze(1)  # [B, 1, len(targets)]

        mask_logits = None
        if self.mask_head is not None:
            mask_logits = self.mask_head(in_memory).view(-1, self._mask_h, self._mask_w)

        return action_output, resnet_inter, attn_weights, mask_logits