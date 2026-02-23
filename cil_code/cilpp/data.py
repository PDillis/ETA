import os
import json
import bisect
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import threading
from tqdm import tqdm

import matplotlib.pyplot as plt


############################################################
# Waypoint projection & mask generation (ported from ETA visutils.py)
############################################################

def project_waypoints_to_pixels(waypoints, config):
    """Project ego-frame waypoints [N,2] (forward, lateral) to front camera pixels [N,2]."""
    K = np.array([
        [config.camera_focal_length, 0.0, config.camera_cx],
        [0.0, config.camera_focal_length, config.camera_cy],
        [0.0, 0.0, 1.0],
    ])
    # ETA convention: ego frame [x_forward, y_lateral] → camera frame [y_lateral, height, x_forward]
    # Negate lateral per ETA data.py:532 (CARLA positive-y = left)
    points_cam = np.stack([
        -waypoints[:, 1],                                    # lateral → camera x (negated)
        np.full(len(waypoints), config.camera_height),       # fixed height
        waypoints[:, 0],                                     # forward → camera z (depth)
    ], axis=-1)

    # Guard against zero/negative depth
    depth = points_cam[:, 2:3].copy()
    mask = np.abs(depth) < 1e-4
    depth[mask] = 1e-4
    points_cam = points_cam / depth

    pixel = (K @ points_cam.T).T
    pixel = -pixel
    pixel[:, 0] += config.camera_original_size[0] // 2
    pixel[:, 1] += config.camera_original_size[1] // 2
    return pixel[:, :2]


def generate_waypoint_mask(waypoints, config):
    """Generate low-res binary mask [mask_height, mask_width] from ego-frame waypoints [N,2]."""
    # Filter out waypoints behind camera (forward distance <= 0)
    valid = waypoints[:, 0] > 0.5
    if not np.any(valid):
        return np.zeros((config.mask_height, config.mask_width), dtype=np.float32)

    pixel_coords = project_waypoints_to_pixels(waypoints[valid], config)

    orig_w, orig_h = config.camera_original_size
    scale_x = config.mask_width / orig_w
    scale_y = config.mask_height / orig_h

    mask = np.zeros((config.mask_height, config.mask_width), dtype=np.float32)
    r = config.mask_waypoint_radius
    for px, py in zip(pixel_coords[:, 0] * scale_x, pixel_coords[:, 1] * scale_y):
        ix, iy = int(round(px)), int(round(py))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    ny, nx = iy + dy, ix + dx
                    if 0 <= ny < config.mask_height and 0 <= nx < config.mask_width:
                        mask[ny, nx] = 1.0
    return mask


############################################################
# Sampling strategy registry
# Each strategy: (buckets: np.ndarray[N, num_buckets]) → np.ndarray[N] float32 weights
# To add a new strategy: define a function and register it in SAMPLING_STRATEGIES.
############################################################

NUM_BEHAVIORAL_BUCKETS = 16   # first 16 columns = behavioral buckets
CMD_BUCKET_START = 16         # columns 16-21 = command buckets
NUM_CMD_BUCKETS = 6


def _weights_uniform(buckets):
    """Normalize per-bucket, then average across all buckets."""
    col_sums = buckets.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    return (buckets / col_sums).mean(axis=-1).astype(np.float32)


def _weights_preferturns(buckets):
    """ETA baseline: weighted average over the 16 behavioral buckets only."""
    MULTIPLIERS = [
        1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0,  # general + accel
        3.0, 3.0,                               # steer right/left
        1.0, 1.0, 1.0,                         # vehicle hazard
        1.0, 1.0, 1.0, 1.0,                    # stop/red/swerve/ped
    ]
    beh = buckets[:, :NUM_BEHAVIORAL_BUCKETS]
    bw = np.array(MULTIPLIERS[:beh.shape[1]])
    col_sums = beh.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    return ((beh / col_sums * bw).sum(axis=-1) / bw.sum()).astype(np.float32)


def _weights_commands(buckets):
    """Inverse-frequency weighting on command columns only — equalizes command distribution."""
    cmd = buckets[:, CMD_BUCKET_START:CMD_BUCKET_START + NUM_CMD_BUCKETS]
    if cmd.shape[1] == 0:
        raise ValueError("No command bucket columns found. Regenerate .npy with command bucket support.")
    cmd_counts = cmd.sum(axis=0)
    cmd_counts[cmd_counts == 0] = 1.0
    return (cmd * (1.0 / cmd_counts)).sum(axis=-1).astype(np.float32)


SAMPLING_STRATEGIES = {
    'uniform': _weights_uniform,
    'preferturns': _weights_preferturns,
    'commands': _weights_commands,
}


############################################################
# Helper: detect whether data_path is a shard directory
############################################################

def _is_shard_path(data_path):
    """Return True if data_path points to a shard directory or manifest.json."""
    if os.path.isdir(data_path):
        return os.path.exists(os.path.join(data_path, 'manifest.json'))
    if os.path.basename(data_path) == 'manifest.json':
        return True
    return False


class CARLA_Data(Dataset):
    """
    Bench2Drive → CIL++ dataset.

    Supports two loading modes:
    - **Monolithic**: Single .npy file loaded entirely into memory (original behavior).
    - **Sharded**: Per-route .npy files loaded on demand via LRU cache (for large datasets).
      Detected when data_path is a directory with manifest.json or a manifest.json file.
      DataLoader shuffle / WeightedDistributedSampler randomize global indices,
      and _resolve_index() maps each to (shard_idx, local_idx) via bisect.
    """

    def __init__(self, root, data_path, config, img_aug=False, split="train", verbose=True):
        self.root = root
        self.config = config
        self.img_aug = img_aug
        self.split = split  # "train" or "val"
        self.verbose = verbose
        self._batch_read_number = 0

        # Camera setup from config
        self.data_used = config.data_used
        self.reference_camera = config.reference_camera

        # Extract image dimensions from config
        self.img_height = config.image_shape[1]
        self.img_width = config.image_shape[2]

        # Dispatch based on data format
        self._shard_mode = _is_shard_path(data_path)
        if self._shard_mode:
            self._init_sharded(data_path, verbose)
        else:
            self._init_monolithic(data_path, verbose)

    # ==================================================================
    # Monolithic loading (original code path)
    # ==================================================================

    def _init_monolithic(self, data_path, verbose):
        """Load entire dataset from a single .npy file into memory."""
        self.front_img = []
        self.x = []
        self.y = []
        self.command = []
        self.target_command = []
        self.target_gps = []
        self.theta = []
        self.speed = []

        self.value = []
        self.feature = []
        self.action = []
        self.action_index = []

        self.future_x = []
        self.future_y = []
        self.future_theta = []

        self.future_feature = []
        self.future_action = []
        self.future_action_index = []
        self.future_only_ap_brake = []

        self.x_command = []
        self.y_command = []
        self.command = []
        self.only_ap_brake = []

        if verbose:
            print(f'Load {self.split} data from {data_path}')
        data = np.load(data_path, allow_pickle=True).item()

        # Toggle to True if you want to warm-load all images to memory
        self.load_to_memory = False
        if self.load_to_memory:
            if verbose:
                print('load data to memory begin')
            self.progress_bar = tqdm(total=len(data['front_img']), desc="Loading Images")
            threads = []
            self.img_cache = {}
            self.lock = threading.Lock()

            for img_path in data['front_img']:
                while threading.active_count() >= 64:
                    for t in threads:
                        t.join()
                    threads = [t for t in threads if t.is_alive()]
                thread = threading.Thread(target=self.load_image, args=(img_path,))
                thread.start()
                threads.append(thread)

            for t in threads:
                t.join()
            self.progress_bar.close()
            if verbose:
                print('load data to memory end')

        # Fill lists
        self.x_command += data['x_target']
        self.y_command += data['y_target']
        self.command += data['target_command']

        self.front_img += data['front_img']
        self.x += data['input_x']
        self.y += data['input_y']
        self.theta += data['input_theta']
        self.speed += data['speed']

        self.future_x += data['future_x']
        self.future_y += data['future_y']
        self.future_theta += data['future_theta']

        self.future_feature += data['future_feature']
        self.future_action += data['future_action']
        self.future_action_index += data['future_action_index']
        self.future_only_ap_brake += data['future_only_ap_brake']

        self.value += data['value']
        self.feature += data['feature']
        self.action += data['action']
        self.action_index += data['action_index']
        self.only_ap_brake += data['only_ap_brake']

        # Bucket vectors for weighted sampling (optional — backward compatible)
        self.buckets = None
        if 'buckets' in data:
            self.buckets = np.array(data['buckets'], dtype=np.float64)

        # Warm-loaded sensor images (optional — backward compatible with path-based .npy)
        self.warm_loaded = data.get('warm_load_size', None) is not None
        self.sensor_list = data.get('sensor_list', None)
        self.sensor_images = {}
        if self.warm_loaded and self.sensor_list:
            for sensor_name in self.sensor_list:
                if sensor_name in data:
                    self.sensor_images[sensor_name] = data[sensor_name]
            if verbose:
                print(f'Warm-loaded sensors: {list(self.sensor_images.keys())} '
                      f'(size: {data["warm_load_size"]})')

        # Sensor metadata (intrinsics/extrinsics, optional)
        self.sensor_meta = data.get('sensor_meta', None)

    # ==================================================================
    # Sharded loading (lazy, LRU-cached)
    # ==================================================================

    def _init_sharded(self, data_path, verbose):
        """Load manifest and build cumulative index for lazy shard loading.

        Instead of loading the entire dataset into memory, we only load
        per-route .npy files on demand. The DataLoader's shuffle (or
        WeightedDistributedSampler) generates random global indices;
        _resolve_index() maps each to the correct (shard, local_idx)
        via binary search on cumulative offsets.

        Peak RAM per rank ≈ shard_cache_size × avg_route_size (~100MB)
        instead of the full dataset (~147GB).
        """
        if os.path.isdir(data_path):
            manifest_path = os.path.join(data_path, 'manifest.json')
            shard_dir = data_path
        else:
            manifest_path = data_path
            shard_dir = os.path.dirname(data_path)

        with open(manifest_path) as f:
            manifest = json.load(f)

        self._shard_dir = shard_dir
        self._total_samples = manifest['total_samples']

        # Build cumulative offset array: _shard_offsets[i] = first global index of shard i
        # _shard_offsets has len(shards)+1 entries (last = total_samples, sentinel)
        self._shard_files = []
        self._shard_offsets = []
        offset = 0
        for route_info in manifest['routes']:
            self._shard_files.append(route_info['file'])
            self._shard_offsets.append(offset)
            offset += route_info['samples']
        self._shard_offsets.append(offset)  # sentinel

        # Load buckets for weighted sampling (~84MB for 475k samples — fits easily)
        buckets_path = os.path.join(shard_dir, '_buckets.npy')
        self.buckets = np.load(buckets_path) if os.path.exists(buckets_path) else None

        # Warm-load metadata (in shard mode, images are inside individual shard files)
        self.warm_loaded = manifest.get('warm_load_size') is not None
        self.sensor_list = manifest.get('sensor_list')
        self.sensor_meta = manifest.get('sensor_meta')
        self.sensor_images = {}  # not used in shard mode

        # LRU cache for loaded shards
        self._shard_cache_size = getattr(self.config, 'shard_cache_size', 20)
        self._shard_cache = {}
        self._shard_lru = []  # ordered list for LRU eviction

        # Compatibility attributes (not used in shard mode)
        self.load_to_memory = False

        if verbose:
            print(f'Sharded dataset: {len(self._shard_files)} shards, '
                  f'{self._total_samples} samples, cache_size={self._shard_cache_size}')

    def _resolve_index(self, global_idx):
        """Map global sample index → (shard_idx, local_idx) via binary search."""
        shard_idx = bisect.bisect_right(self._shard_offsets, global_idx) - 1
        local_idx = global_idx - self._shard_offsets[shard_idx]
        return shard_idx, local_idx

    def _load_shard(self, shard_idx):
        """Load a shard into the LRU cache, evicting oldest if full."""
        if shard_idx in self._shard_cache:
            # Move to most-recently-used position
            self._shard_lru.remove(shard_idx)
            self._shard_lru.append(shard_idx)
            return self._shard_cache[shard_idx]

        # Evict oldest if cache is full
        while len(self._shard_lru) >= self._shard_cache_size:
            evict_idx = self._shard_lru.pop(0)
            del self._shard_cache[evict_idx]

        # Load shard from disk
        path = os.path.join(self._shard_dir, self._shard_files[shard_idx])
        shard = np.load(path, allow_pickle=True).item()
        shard.pop('skipped_nan', None)
        self._shard_cache[shard_idx] = shard
        self._shard_lru.append(shard_idx)
        return shard

    # ==================================================================
    # Shared interface
    # ==================================================================

    def get_sample_weights(self):
        """Compute per-sample weights from bucket vectors. Returns (N,) float32 array."""
        if self.buckets is None:
            raise ValueError("No bucket data available. Regenerate .npy with bucket support.")
        strategy = self.config.bucket_weight_type
        if strategy not in SAMPLING_STRATEGIES:
            available = ', '.join(sorted(SAMPLING_STRATEGIES.keys()))
            raise ValueError(f"Unknown sampling strategy '{strategy}'. Available: {available}")
        return SAMPLING_STRATEGIES[strategy](self.buckets)

    def load_image(self, img_path):
        """Load all configured camera views for a given reference image path into the cache."""
        for cam_name in self.data_used:
            cam_path = img_path.replace(self.reference_camera, cam_name)
            img = np.array(Image.open(cam_path))
            with self.lock:
                self.img_cache[cam_path] = img
        with self.lock:
            self.progress_bar.update(1)

    def __len__(self):
        if self._shard_mode:
            return self._total_samples
        return len(self.front_img)

    def __getitem__(self, index):
        if self._shard_mode:
            return self._getitem_sharded(index)
        return self._getitem_monolithic(index)

    # ==================================================================
    # __getitem__ — monolithic path (original code)
    # ==================================================================

    def _getitem_monolithic(self, index):
        data = dict()

        # ---------- Load camera views ----------
        base_path = self.front_img[index][0]
        if not os.path.exists(base_path):
            base_path = base_path.replace('v2', 'v2-216')

        resize_dims = (self.img_width, self.img_height)
        for cam_name in self.data_used:
            if self.warm_loaded and cam_name in self.sensor_images:
                # Already resized at generation time — skip resize
                cam_img = self._ensure_pil(self.sensor_images[cam_name][index])
            else:
                cam_path = base_path.replace(self.reference_camera, cam_name)
                if self.load_to_memory:
                    cam_img = self.img_cache[cam_path]
                else:
                    cam_img = np.array(Image.open(cam_path))
                cam_img = self._ensure_pil(cam_img).resize(resize_dims, Image.BILINEAR)
            data[cam_name] = cam_img

        # ---------- Normalize like CIL++ ----------
        if self.split == "train":
            data = self.train_transform(data, augmentation=self.img_aug)
        else:
            data = self.val_transform(data)

        # You can add more fields if needed later; for image-debug we stop here
        ego_x = self.x[index][0]
        ego_y = self.y[index][0]
        ego_theta = self.theta[index][0] - np.pi/2 # compass on left hand (0, -1)
        waypoints = []
        R = np.array([
            [np.cos(ego_theta), np.sin(ego_theta)],
            [-np.sin(ego_theta),  np.cos(ego_theta)]
            ])
        for i in range(8):
            local_command_point = np.array([self.future_x[index][i]-ego_x, self.future_y[index][i]-ego_y])
            local_command_point = R.dot(local_command_point) # left hand
            waypoints.append([local_command_point[0], local_command_point[1]])

        data['waypoints'] = np.array(waypoints)

        if getattr(self.config, 'mask_loss_enabled', False):
            data['waypoint_mask'] = torch.from_numpy(
                generate_waypoint_mask(data['waypoints'], self.config)
            )

        data['action'] = self.action[index]
        data['action_index'] = self.action_index[index]

        data['future_action_index'] = self.future_action_index[index]
        data['future_feature'] = self.future_feature[index]

        local_command_point_aim = np.array([(self.x_command[index]-ego_x), self.y_command[index]-ego_y])
        local_command_point_aim = R.dot(local_command_point_aim)
        data['target_point'] = local_command_point_aim[:2]

        data['speed'] = self.speed[index]
        data['feature'] = self.feature[index]
        data['value'] = self.value[index]
        command = self.command[index]

        # VOID = -1 ;set to LANEFOLLOW (4) for one-hot encoding
        # LEFT = 1
        # RIGHT = 2
        # STRAIGHT = 3
        # LANEFOLLOW = 4
        # CHANGELANELEFT = 5
        # CHANGELANERIGHT = 6
        if command < 0:
            command = 4
        command -= 1
        assert command in [0, 1, 2, 3, 4, 5]
        cmd_one_hot = [0] * self.config.data_command_class_num
        cmd_one_hot[command] = 1
        data['target_command'] = torch.tensor(cmd_one_hot)

        self._batch_read_number += 1

        return data

    # ==================================================================
    # __getitem__ — sharded path (lazy loading)
    # ==================================================================

    def _getitem_sharded(self, index):
        shard_idx, local_idx = self._resolve_index(index)
        shard = self._load_shard(shard_idx)
        data = dict()

        # ---------- Load camera views ----------
        base_path = shard['front_img'][local_idx][0]
        if not os.path.exists(base_path):
            base_path = base_path.replace('v2', 'v2-216')

        resize_dims = (self.img_width, self.img_height)
        for cam_name in self.data_used:
            if self.warm_loaded and cam_name in shard:
                cam_img = self._ensure_pil(shard[cam_name][local_idx])
            else:
                cam_path = base_path.replace(self.reference_camera, cam_name)
                cam_img = np.array(Image.open(cam_path))
                cam_img = self._ensure_pil(cam_img).resize(resize_dims, Image.BILINEAR)
            data[cam_name] = cam_img

        # ---------- Normalize like CIL++ ----------
        if self.split == "train":
            data = self.train_transform(data, augmentation=self.img_aug)
        else:
            data = self.val_transform(data)

        # Waypoints (same ego-frame transform as monolithic path)
        ego_x = shard['input_x'][local_idx][0]
        ego_y = shard['input_y'][local_idx][0]
        ego_theta = shard['input_theta'][local_idx][0] - np.pi / 2
        R = np.array([
            [np.cos(ego_theta), np.sin(ego_theta)],
            [-np.sin(ego_theta), np.cos(ego_theta)]
        ])
        waypoints = []
        for i in range(8):
            lp = np.array([shard['future_x'][local_idx][i] - ego_x,
                           shard['future_y'][local_idx][i] - ego_y])
            waypoints.append(R.dot(lp))
        data['waypoints'] = np.array(waypoints)

        if getattr(self.config, 'mask_loss_enabled', False):
            data['waypoint_mask'] = torch.from_numpy(
                generate_waypoint_mask(data['waypoints'], self.config)
            )

        data['action'] = shard['action'][local_idx]
        data['action_index'] = shard['action_index'][local_idx]
        data['future_action_index'] = shard['future_action_index'][local_idx]
        data['future_feature'] = shard['future_feature'][local_idx]

        local_aim = np.array([shard['x_target'][local_idx] - ego_x,
                              shard['y_target'][local_idx] - ego_y])
        data['target_point'] = R.dot(local_aim)[:2]

        data['speed'] = shard['speed'][local_idx]
        data['feature'] = shard['feature'][local_idx]
        data['value'] = shard['value'][local_idx]

        command = shard['target_command'][local_idx]
        if command < 0:
            command = 4
        command -= 1
        assert command in [0, 1, 2, 3, 4, 5]
        cmd_one_hot = [0] * self.config.data_command_class_num
        cmd_one_hot[command] = 1
        data['target_command'] = torch.tensor(cmd_one_hot)

        self._batch_read_number += 1

        return data

    ###############################
    #          HELPERS
    ###############################

    @staticmethod
    def _ensure_pil(img_np_or_pil):
        if isinstance(img_np_or_pil, np.ndarray):
            return Image.fromarray(img_np_or_pil)
        elif isinstance(img_np_or_pil, Image.Image):
            return img_np_or_pil
        else:
            raise TypeError(f"Unexpected image type: {type(img_np_or_pil)}")

    def train_transform(self, data, augmentation=False):
        for camera_type in self.data_used:
            img = data[camera_type]
            # images already resized in __getitem__
            t = TF.to_tensor(img)
            t = TF.normalize(t,
                            mean=self.config.img_mean,
                            std=self.config.img_std)
            data[camera_type] = t
        return data

    def val_transform(self, data):
        for camera_type in self.data_used:
            img = data[camera_type]
            t = TF.to_tensor(img)
            t = TF.normalize(t,
                            mean=self.config.img_mean,
                            std=self.config.img_std)
            data[camera_type] = t
        return data
