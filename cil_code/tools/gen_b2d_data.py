#!/usr/bin/env python3
"""
Configurable dataset creation for imitation learning.

This script generates .npy training/validation data with configurable:
- Dataset format (Bench2Drive, Play2Drive, etc. via adapter pattern)
- Framerate conversion (input_fps → output_fps)
- Sequence lengths (input_frames, future_frames)
- Train/val split (via route lists or config)

Usage:
    # Auto-named output (recommended): produces b2d-base_2Hz_1Input_8Future-train.npy
    python gen_b2d_data.py \\
        --dataset-root /path/to/b2d-base \\
        --format bench2drive \\
        --train

    # With explicit base name: produces mydata_2Hz_1Input_8Future-val.npy
    python gen_b2d_data.py \\
        --dataset-root /path/to/b2d-base \\
        --name mydata \\
        --val

    # With explicit output path (overrides auto-name)
    python gen_b2d_data.py \\
        --dataset-root /path/to/b2d-base \\
        --output custom_output.npy

    # Using config file (with CLI overrides)
    python gen_b2d_data.py \\
        --config tools/bench2drive_config.yaml \\
        --train
"""

import os
import sys
import json
import yaml
import click
import numpy as np
import multiprocessing as mp
from typing import Optional, List, Dict, Any
from pathlib import Path
from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn,
    MofNCompleteColumn, SpinnerColumn,
)
from rich.table import Table
from rich import box

console = Console()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.dataset_formats import Bench2DriveAdapter, Play2DriveAdapter


def make_output_name(name: str, output_fps: int, input_frames: int, future_frames: int, split: str) -> str:
    """
    Auto-generate an output filename from dataset parameters.

    Format: {name}_{output_fps}Hz_{input_frames}Input_{future_frames}Future-{split}.npy

    Examples:
        b2d-base_2Hz_1Input_8Future-train.npy
        play2drive_10Hz_2Input_8Future-val.npy
    """
    return f"{name}_{output_fps}Hz_{input_frames}Input_{future_frames}Future-{split}.npy"


############################################################
# Hazard helpers (ported from ETA: carformer/data/data_utils.py)
############################################################

def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1
    if abs(np.linalg.det(A)) < 1e-3:
        return False, None
    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 4)  # collision within 4 seconds
    return collides, p1 + x[0] * v1


def get_hazard_directions(vehicle_list):
    """Return list of angle_from_ego for nearby vehicle hazards."""
    ego_vehicles = [x for x in vehicle_list if x["class"] == "ego_vehicle"]
    if len(ego_vehicles) != 1:
        return []

    ego_vehicle = ego_vehicles[0]
    o1 = _orientation(ego_vehicle["rotation"][-1])
    p1 = np.asarray(ego_vehicle["location"][:2])
    s1 = max(2, 3.0 * ego_vehicle["speed"])
    v1_hat = o1

    hazard_directions = []
    for target_vehicle in vehicle_list:
        if target_vehicle["class"] == "ego_vehicle":
            continue
        if target_vehicle.get("base_type", None) != "car":
            continue

        o2 = _orientation(target_vehicle["rotation"][-1])
        p2 = np.asarray(target_vehicle["location"][:2])

        p2_p1 = p2 - p1
        distance = np.linalg.norm(p2_p1)
        p2_p1_hat = p2_p1 / (distance + 1e-4)

        angle_to_car = np.degrees(np.arccos(np.clip(v1_hat.dot(p2_p1_hat), -1, 1)))
        angle_between_heading = np.degrees(np.arccos(np.clip(o1.dot(o2), -1, 1)))
        angle_from_ego = np.degrees(np.arccos(np.clip(o2.dot(p2_p1_hat), -1, 1)))

        angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
        angle_between_heading = min(angle_between_heading, 360.0 - angle_between_heading)

        if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
            continue
        elif angle_to_car > 30.0:
            continue
        elif distance > s1:
            continue

        hazard_directions.append(angle_from_ego)

    return hazard_directions


def is_walker_hazard(objects_list):
    """Return True if any walker is on a collision course with ego within 4 seconds."""
    ego_vehicles = [x for x in objects_list if x["class"] == "ego_vehicle"]
    if len(ego_vehicles) == 0:
        return False

    ego_vehicle = ego_vehicles[0]
    p1 = np.asarray(ego_vehicle["location"][:2])
    v1 = 10.0 * _orientation(ego_vehicle["rotation"][-1])

    walkers = [x for x in objects_list if x["class"] == "walker"]
    for walker in walkers:
        v2_hat = _orientation(walker["rotation"][-1])
        s2 = walker["speed"]
        if s2 < 0.05:
            v2_hat *= s2
        p2 = -3.0 * v2_hat + np.asarray(walker["location"][:2])
        v2 = 8.0 * v2_hat
        collides, _ = get_collision(p1, v1, p2, v2)
        if collides:
            return True
    return False


############################################################
# Data bucketing (ported from ETA: carformer/data/data_parser.py)
############################################################

BUCKET_NAMES = [
    # Behavioral buckets (0-15)
    'general', 'acc_scratch', 'acc_light_pedal', 'acc_medium_pedal',
    'acc_heavy_pedal', 'acc_brake', 'acc_coast', 'steer_right', 'steer_left',
    'vehicle_hazard_front', 'vehicle_hazard_back', 'vehicle_hazard_side',
    'stop_sign', 'red_light', 'swerving', 'pedestrian',
    # Command buckets (16-21, mutually exclusive)
    'cmd_left', 'cmd_right', 'cmd_straight',
    'cmd_lanefollow', 'cmd_changelaneleft', 'cmd_changelaneright',
]

SWERVING_SCENARIOS = [
    "Accident", "BlockedIntersection", "ConstructionObstacle",
    "HazardAtSideLane", "ParkedObstacle", "VehicleOpensDoorTwoWays",
]


def compute_buckets(throttle, steer, brake, speed, bounding_boxes, route_name, command):
    """Compute 22-element binary bucket vector for a single sample."""
    # Acceleration buckets (thresholds from ETA data_parser.py)
    acc_bucket = [
        1 if (throttle > 0.2 and brake < 1.0 and speed < 0.05) else 0,  # scratch
        1 if (throttle > 0.2 and throttle < 0.5) else 0,                # light pedal
        1 if (throttle > 0.5 and throttle < 0.9) else 0,                # medium pedal
        1 if (throttle > 0.9) else 0,                                    # heavy pedal
        1 if (brake > 0.2) else 0,                                       # brake
        1 if (throttle < 0.2 and brake < 1.0) else 0,                   # coast
    ]
    steer_bucket = [1 if steer > 0.2 else 0, 1 if steer < -0.2 else 0]

    # Vehicle hazard (from bounding_boxes)
    hazard_angles = get_hazard_directions(bounding_boxes)
    veh_bucket = [
        1 if any(a < 30 for a in hazard_angles) else 0,       # front
        1 if any(a > 150 for a in hazard_angles) else 0,      # back
        1 if any(30 < a < 150 for a in hazard_angles) else 0, # side
    ]

    # Stop sign
    stopsigns = [x for x in bounding_boxes
                 if x.get("class") == "traffic_sign" and x.get("type_id") == "traffic.stop"]
    stop_bucket = 1 if any(x.get("affects_ego", False) for x in stopsigns) else 0

    # Red light
    redlights = [x for x in bounding_boxes
                 if x.get("class") == "traffic_light" and x.get("state") == 0]
    red_bucket = 1 if any(x.get("affects_ego", False) for x in redlights) else 0

    # Swerving
    is_swerving_route = any(s in route_name for s in SWERVING_SCENARIOS)
    swerve_bucket = 1 if (is_swerving_route and abs(steer) > 0.1) else 0

    # Pedestrian
    ped_bucket = 1 if is_walker_hazard(bounding_boxes) else 0

    # Command buckets (VOID=-1 → LANEFOLLOW=4, same convention as data.py)
    cmd = command if command > 0 else 4
    cmd_bucket = [1 if cmd == c else 0 for c in [1, 2, 3, 4, 5, 6]]

    return [1] + acc_bucket + steer_bucket + veh_bucket + [stop_bucket, red_bucket, swerve_bucket, ped_bucket] + cmd_bucket


def _all_finite(*xs) -> bool:
    """Check if all values are finite (no NaN/Inf)."""
    for x in xs:
        if isinstance(x, (str, bool)):
            continue
        a = np.asarray(x, dtype=np.float32)
        if not np.all(np.isfinite(a)):
            return False
    return True


def process_single_route(
    route_folder: str,
    adapter,
    config: Dict[str, Any],
    count,
    worker_id: int = 0,
    progress_queue: Optional[mp.Queue] = None,
) -> Optional[Dict]:
    """
    Process a single route directory and extract sequences.

    Args:
        route_folder: Path to route directory
        adapter: Dataset format adapter
        config: Configuration dictionary with keys:
                - input_frames: int
                - future_frames: int
                - step: int (framerate conversion factor)
        count: Multiprocessing counter for progress tracking
        worker_id: Worker index (for Rich progress display)
        progress_queue: Queue to send progress messages to the main process

    Returns:
        Dictionary with sequence data or None if route is too short
    """
    input_frames = config['input_frames']
    future_frames_raw = config['future_frames'] * config['step']  # At input FPS
    step = config['step']

    # Determine route length
    anno_dir = os.path.dirname(adapter.get_annotation_path(route_folder, 0))
    if not os.path.isdir(anno_dir):
        return None

    length = len([name for name in os.listdir(anno_dir)]) - 1  # Drop last frame

    if length < input_frames + future_frames_raw:
        return None

    # Sequence lists
    seq_future_x, seq_future_y, seq_future_theta = [], [], []
    seq_future_feature, seq_future_action, seq_future_action_index = [], [], []
    seq_future_only_ap_brake = []
    seq_future_acceleration = []
    seq_future_angular_velocity = []

    seq_input_x, seq_input_y, seq_input_theta = [], [], []
    seq_front_img, seq_feature, seq_value, seq_speed = [], [], [], []
    seq_action, seq_action_index = [], []
    seq_x_target, seq_y_target, seq_target_command = [], [], []
    seq_only_ap_brake = []
    seq_acceleration = []
    seq_angular_velocity = []
    seq_buckets = []

    # Full sequences (for downsampling)
    full_seq_x, full_seq_y, full_seq_theta = [], [], []
    full_seq_feature, full_seq_action, full_seq_action_index = [], [], []
    full_seq_only_ap_brake = []
    full_seq_acceleration = []
    full_seq_angular_velocity = []

    # Load all frames at input FPS
    basename = os.path.basename(route_folder)
    for i in range(length):
        anno_path = adapter.get_annotation_path(route_folder, i)
        anno = adapter.load_annotation(anno_path)

        # Load expert assessment (Bench2Drive-specific)
        if hasattr(adapter, 'get_expert_assessment_path'):
            expert_path = adapter.get_expert_assessment_path(route_folder, i)
            expert_feature = np.load(expert_path, allow_pickle=True)['arr_0']
        else:
            # For datasets without expert assessment, extract from annotation
            # TODO: Adjust based on actual Play2Drive format
            raise NotImplementedError("Non-Bench2Drive datasets not yet supported")

        full_seq_x.append(anno['x'])
        full_seq_y.append(anno['y'])
        full_seq_theta.append(anno['theta'])
        full_seq_feature.append(expert_feature[:-2])
        throttle, steer, brake, _ = adapter.get_action(int(expert_feature[-1]))
        full_seq_action.append(np.array([throttle, steer, brake], dtype=np.float32))
        full_seq_action_index.append(int(expert_feature[-1]))
        full_seq_only_ap_brake.append(anno['only_ap_brake'])
        full_seq_acceleration.append(anno.get('acceleration', [0.0, 0.0, 0.0]))
        full_seq_angular_velocity.append(anno.get('angular_velocity', [0.0, 0.0, 0.0]))

    _skipped_nan = 0

    # Extract sequences at output FPS
    sample_start = input_frames - 1
    sample_end = length - future_frames_raw - step
    for i in range(sample_start, sample_end):
        anno_path = adapter.get_annotation_path(route_folder, i)
        anno = adapter.load_annotation(anno_path)

        if hasattr(adapter, 'get_expert_assessment_path'):
            expert_path = adapter.get_expert_assessment_path(route_folder, i)
            expert_feature = np.load(expert_path, allow_pickle=True)['arr_0']
        else:
            raise NotImplementedError("Non-Bench2Drive datasets not yet supported")

        # Downsample sequences to output FPS
        inp_x_seq = full_seq_x[i - (input_frames - 1):i + step:step]
        inp_y_seq = full_seq_y[i - (input_frames - 1):i + step:step]
        inp_th_seq = full_seq_theta[i - (input_frames - 1):i + step:step]

        fut_x_seq = full_seq_x[i + step:i + future_frames_raw + step:step]
        fut_y_seq = full_seq_y[i + step:i + future_frames_raw + step:step]
        fut_th_seq = full_seq_theta[i + step:i + future_frames_raw + step:step]

        fut_feat_seq = full_seq_feature[i + step:i + future_frames_raw + step:step]
        fut_act_seq = full_seq_action[i + step:i + future_frames_raw + step:step]
        fut_act_idx_seq = full_seq_action_index[i + step:i + future_frames_raw + step:step]
        fut_only_ap = full_seq_only_ap_brake[i + step:i + future_frames_raw + step:step]
        fut_accel_seq = full_seq_acceleration[i + step:i + future_frames_raw + step:step]
        fut_ang_vel_seq = full_seq_angular_velocity[i + step:i + future_frames_raw + step:step]

        cur_feat = expert_feature[:-2]
        cur_value = expert_feature[-2]
        cur_speed = anno["speed"]
        cur_xt, cur_yt = anno["x_target"], anno["y_target"]
        cur_accel = anno.get("acceleration", [0.0, 0.0, 0.0])  # [x, y, z]
        cur_ang_vel = anno.get("angular_velocity", [0.0, 0.0, 0.0])  # [x, y, z]

        # NaN/Inf filtering
        if not _all_finite(inp_x_seq, inp_y_seq, inp_th_seq,
                           fut_x_seq, fut_y_seq, fut_th_seq,
                           cur_feat, cur_value, cur_speed, cur_xt, cur_yt,
                           cur_accel, cur_ang_vel):
            _skipped_nan += 1
            continue
        if not _all_finite(np.asarray(fut_feat_seq, dtype=np.float32)):
            _skipped_nan += 1
            continue
        if not _all_finite(np.asarray(fut_accel_seq, dtype=np.float32)):
            _skipped_nan += 1
            continue
        if not _all_finite(np.asarray(fut_ang_vel_seq, dtype=np.float32)):
            _skipped_nan += 1
            continue

        seq_input_x.append(inp_x_seq)
        seq_input_y.append(inp_y_seq)
        seq_input_theta.append(inp_th_seq)

        seq_future_x.append(fut_x_seq)
        seq_future_y.append(fut_y_seq)
        seq_future_theta.append(fut_th_seq)

        seq_future_feature.append(fut_feat_seq)
        seq_future_action.append(fut_act_seq)
        seq_future_action_index.append(fut_act_idx_seq)
        seq_future_only_ap_brake.append(fut_only_ap)
        seq_future_acceleration.append(fut_accel_seq)
        seq_future_angular_velocity.append(fut_ang_vel_seq)

        seq_feature.append(cur_feat)
        seq_value.append(cur_value)

        # Image paths (reference camera only)
        ref_camera = adapter.get_reference_camera()
        seq_front_img.append([
            adapter.get_image_path(route_folder, ref_camera, i)
            for _ in range(input_frames - 1, -1, -1)
        ])
        seq_speed.append(cur_speed)

        throttle, steer, brake, _ = adapter.get_action(int(expert_feature[-1]))
        seq_action.append(np.array([throttle, steer, brake], dtype=np.float32))
        seq_action_index.append(int(expert_feature[-1]))

        seq_x_target.append(cur_xt)
        seq_y_target.append(cur_yt)
        seq_target_command.append(anno["next_command"])
        seq_only_ap_brake.append(anno["only_ap_brake"])
        seq_acceleration.append(cur_accel)
        seq_angular_velocity.append(cur_ang_vel)

        # Bucket computation
        bounding_boxes = anno.get('bounding_boxes', [])
        bucket = compute_buckets(throttle, steer, brake, cur_speed, bounding_boxes, basename, anno["next_command"])
        seq_buckets.append(bucket)

    with count.get_lock():
        count.value += 1

    if progress_queue is not None:
        progress_queue.put({
            'type': 'route_done',
            'route': basename,
            'group': adapter.get_route_group(basename),
            'skipped': _skipped_nan,
        })

    return {
        'future_x': seq_future_x,
        'future_y': seq_future_y,
        'future_theta': seq_future_theta,
        'future_feature': seq_future_feature,
        'future_action': seq_future_action,
        'future_action_index': seq_future_action_index,
        'future_only_ap_brake': seq_future_only_ap_brake,
        'future_acceleration': seq_future_acceleration,
        'future_angular_velocity': seq_future_angular_velocity,
        'input_x': seq_input_x,
        'input_y': seq_input_y,
        'input_theta': seq_input_theta,
        'front_img': seq_front_img,
        'feature': seq_feature,
        'value': seq_value,
        'speed': seq_speed,
        'action': seq_action,
        'action_index': seq_action_index,
        'x_target': seq_x_target,
        'y_target': seq_y_target,
        'target_command': seq_target_command,
        'only_ap_brake': seq_only_ap_brake,
        'acceleration': seq_acceleration,
        'angular_velocity': seq_angular_velocity,
        'buckets': seq_buckets,
        'skipped_nan': _skipped_nan,
    }


def worker(
    folder_paths: mp.Queue,
    count,
    seq_data_list,
    adapter,
    config,
    worker_id: int = 0,
    progress_queue: Optional[mp.Queue] = None,
):
    """Worker process for parallel route processing."""
    while True:
        if folder_paths.qsize() <= 0:
            break
        folder_path = folder_paths.get()
        seq_data = process_single_route(folder_path, adapter, config, count, worker_id, progress_queue)
        if seq_data is not None:
            seq_data_list.append(seq_data)

    if progress_queue is not None:
        progress_queue.put({'type': 'worker_done', 'worker_id': worker_id})


def run_rich_display(progress_queue: mp.Queue, num_workers: int, group_counts: Dict[str, int]):
    """
    Run a Rich progress display in the main process, one bar per scenario group.

    Blocks until all workers have sent a 'worker_done' message.
    Completed groups are removed from the live panel and logged above it.

    Args:
        progress_queue: Queue receiving route_done / worker_done messages from workers
        num_workers: Total number of worker processes (for termination detection)
        group_counts: {group_name: total_routes_in_group} pre-computed from route_folders
    """
    columns = [
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ]
    total_routes = sum(group_counts.values())
    with Progress(*columns, console=console) as progress:
        # One task per group, sorted alphabetically
        group_tasks = {
            group: progress.add_task(f"[cyan]{group}[/cyan]", total=count)
            for group, count in sorted(group_counts.items())
        }
        overall = progress.add_task("[bold]Total[/bold]", total=total_routes)

        # Track completions ourselves to avoid indexing into progress.tasks by TaskID
        group_completed = {group: 0 for group in group_counts}

        done_workers = 0
        while done_workers < num_workers:
            msg = progress_queue.get()

            if msg['type'] == 'route_done':
                group = msg['group']
                skipped = msg['skipped']
                skip_str = f"  [dim]({skipped} NaN skipped)[/dim]" if skipped else ""
                progress.console.log(f"[green]✓[/green] {msg['route']}{skip_str}")

                group_completed[group] += 1
                progress.update(group_tasks[group], advance=1)
                progress.update(overall, advance=1)

                # Remove completed groups from the live panel
                if group_completed[group] >= group_counts[group]:
                    n = group_counts[group]
                    progress.console.log(
                        f"[bold green]✓✓[/bold green] {group}  —  {n}/{n} routes done"
                    )
                    progress.remove_task(group_tasks[group])

            elif msg['type'] == 'worker_done':
                done_workers += 1


def aggregate_and_save(seq_data_list: List[Dict], output_path: str):
    """Aggregate data from all routes and save to .npy file."""
    console.print('\n[bold]Aggregating and saving...[/bold]')

    total_data = {
        'future_x': [], 'future_y': [], 'future_theta': [],
        'future_feature': [], 'future_action': [], 'future_action_index': [], 'future_only_ap_brake': [],
        'future_acceleration': [], 'future_angular_velocity': [],
        'input_x': [], 'input_y': [], 'input_theta': [],
        'front_img': [], 'feature': [], 'value': [], 'speed': [],
        'action': [], 'action_index': [],
        'x_target': [], 'y_target': [], 'target_command': [],
        'only_ap_brake': [],
        'acceleration': [],
        'angular_velocity': [],
        'buckets': [],
    }

    _skipped_nan_total = 0

    for seq_data in seq_data_list:
        if not seq_data:
            continue
        skipped_nan = seq_data.pop('skipped_nan')
        _skipped_nan_total += skipped_nan

        for key in total_data.keys():
            total_data[key].extend(seq_data[key])

    total_data['bucket_names'] = BUCKET_NAMES
    np.save(output_path, total_data)
    console.print(f'[green]Saved {len(total_data["front_img"])} sequences to {output_path}[/green]')
    console.print(f'[yellow]Total sequences skipped due to NaN/Inf: {_skipped_nan_total}[/yellow]')


@click.command()
@click.option('--config', help='Config file (YAML/JSON)', type=click.Path(exists=True), default=None)
@click.option('--dataset-root', help='Dataset root directory', type=click.Path(exists=True), required=True)
@click.option('--name', help='Base name for auto-generated output filename (default: dataset root basename)', type=str, default=None)
@click.option('--output', help='Output .npy path (overrides auto-generated name)', type=str, default=None)
@click.option('--train/--val', 'is_train', help='Train or validation mode', default=True, show_default=True)
@click.option('--input-fps', help='Input framerate (Hz)', type=int, default=10, show_default=True)
@click.option('--output-fps', help='Output framerate (Hz)', type=int, default=2, show_default=True)
@click.option('--input-frames', help='Total input frames (1 = current only, 2 = 1 history + current, ...)', type=int, default=1, show_default=True)
@click.option('--future-frames', help='Future waypoints at output FPS', type=int, default=8, show_default=True)
@click.option('--val-routes', help='Validation route file (one per line)', type=click.Path(exists=True), default=None)
@click.option('--format', 'dataset_format', help='Dataset format', type=click.Choice(['bench2drive', 'play2drive']), default='bench2drive', show_default=True)
@click.option('--num-workers', help='Number of parallel workers', type=int, default=64, show_default=True)
def main(config, dataset_root, name, output, is_train, input_fps, output_fps, input_frames, future_frames, val_routes, dataset_format, num_workers):
    """
    Generate .npy training/validation data with configurable parameters.

    This script uses the adapter pattern to support multiple dataset formats
    (Bench2Drive, Play2Drive, etc.) with different directory structures and
    annotation formats.
    """
    # TODO: Remove frames where reverse=True
    # Load config file if provided
    cfg = {}
    if config is not None:
        with open(config, 'r') as f:
            if config.endswith('.yaml') or config.endswith('.yml'):
                cfg = yaml.safe_load(f)
            elif config.endswith('.json'):
                cfg = json.load(f)
            else:
                raise click.BadParameter("Config file must be .yaml or .json")

    # CLI args override config file
    dataset_root = dataset_root or cfg.get('dataset', {}).get('root')
    is_train = is_train if is_train is not None else cfg.get('split', {}).get('mode') == 'train'
    input_fps = input_fps or cfg.get('sampling', {}).get('input_fps', 10)
    output_fps = output_fps or cfg.get('sampling', {}).get('output_fps', 2)
    input_frames = input_frames or cfg.get('sampling', {}).get('input_frames', 1)
    future_frames = future_frames or cfg.get('sampling', {}).get('future_frames', 8)
    val_routes = val_routes or cfg.get('split', {}).get('val_routes_file')
    dataset_format = dataset_format or cfg.get('dataset', {}).get('format', 'bench2drive')

    # Validate inputs
    if dataset_root is None:
        raise click.UsageError("--dataset-root is required")
    if input_fps % output_fps != 0:
        raise click.BadParameter(f"input_fps ({input_fps}) must be divisible by output_fps ({output_fps})")

    step = input_fps // output_fps
    split = 'train' if is_train else 'val'

    # Resolve output path: explicit --output overrides auto-generated name
    if output is None:
        output = output or cfg.get('output', {}).get('path')
    if output is None:
        base_name = name or os.path.basename(dataset_root.rstrip('/'))
        output = make_output_name(base_name, output_fps, input_frames, future_frames, split)

    # Load validation routes if provided
    val_route_list = None
    if val_routes is not None:
        with open(val_routes, 'r') as f:
            val_route_list = [line.strip() for line in f if line.strip()]

    # Create adapter
    if dataset_format == 'bench2drive':
        adapter = Bench2DriveAdapter(val_routes=val_route_list)
    elif dataset_format == 'play2drive':
        adapter = Play2DriveAdapter()
    else:
        raise click.BadParameter(f"Unknown dataset format: {dataset_format}")

    # Get route directories and pre-compute scenario group counts for the display
    route_folders = adapter.list_routes(dataset_root, split, val_routes=val_route_list)
    group_counts: Dict[str, int] = {}
    for rf in route_folders:
        g = adapter.get_route_group(os.path.basename(rf))
        group_counts[g] = group_counts.get(g, 0) + 1

    cfg_table = Table(title='Dataset Configuration', box=box.ROUNDED, show_header=False, padding=(0, 1))
    cfg_table.add_column(style='bold cyan')
    cfg_table.add_column()
    rows = [
        ('Format', dataset_format),
        ('Root', dataset_root),
        ('Split', split),
        ('Routes', str(len(route_folders))),
        ('Input FPS', f'{input_fps} Hz'),
        ('Output FPS', f'{output_fps} Hz  (step={step})'),
        ('Input Frames', str(input_frames)),
        ('Future Frames', f'{future_frames}  (at {output_fps} Hz)'),
        ('Output', output),
        ('Num Workers', str(num_workers)),
    ]
    for key, val in rows:
        cfg_table.add_row(key, val)
    console.print(cfg_table)
    console.print()

    # Prepare multiprocessing
    folder_paths = mp.Queue()
    seq_data_list = mp.Manager().list()
    count = mp.Value('d', 0)
    progress_queue = mp.Queue()

    for route_folder in route_folders:
        folder_paths.put(route_folder)

    # Start workers
    ps = []
    processing_config = {
        'input_frames': input_frames,
        'future_frames': future_frames,
        'step': step,
    }

    for i in range(num_workers):
        p = mp.Process(
            target=worker,
            args=(folder_paths, count, seq_data_list, adapter, processing_config, i, progress_queue),
        )
        p.daemon = True
        p.start()
        ps.append(p)

    # Run Rich display in the main process (blocks until all workers done)
    run_rich_display(progress_queue, num_workers, group_counts)

    for p in ps:
        p.join()

    # Aggregate and save
    aggregate_and_save(seq_data_list, output)


if __name__ == '__main__':
    main()