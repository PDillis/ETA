# ETA Mask Loss & Data Bucketing Analysis

Analysis of two key techniques from the ETA paper for integration into CIL++.

## 1. Mask Loss

### Overview

ETA uses an auxiliary mask loss that projects future waypoints onto the front camera view,
creating a binary mask showing where the ego vehicle will travel. This teaches the model
spatial awareness as a secondary supervision signal alongside the primary action loss.

### Projection Math

**Source**: `carformer/carformer/visualization/visutils.py:167` — `point_to_canvas_coordinates_rel_to_center()`

Camera intrinsics (hardcoded for Bench2Drive):
- Focal length: fx = fy = 1142.5184053936916
- Principal point: (cx=800.0, cy=450.0)
- Camera height: -1.6m relative to ground
- Original image: 1600×900 pixels

Projection steps:
1. **Coordinate reorder**: Ego frame `(x_forward, y_lateral)` → camera frame `(y_lateral, height, x_forward)` mapping to `(right, down, forward)` in camera coords
2. **Depth guard**: Clamp near-zero depth to 1e-4
3. **Perspective divide**: `points / depth`
4. **Intrinsic matrix**: `K @ points.T`
5. **Sign flip + re-center**: Negate, then offset by half the image dimensions

```python
# From visutils.py:167-195
K = [[1142.52, 0, 800], [0, 1142.52, 450], [0, 0, 1]]
points_cam = [y_lateral, height, x_forward]  # reorder from ego frame
points_cam /= points_cam[:, 2:]              # perspective divide
pixel_coords = (K @ points_cam.T).T          # project
pixel_coords = -pixel_coords                 # negate
pixel_coords[:, 0] += 800                    # re-center x
pixel_coords[:, 1] += 450                    # re-center y
```

### Mask Generation

**Source**: `carformer/carformer/data/data.py:518-566`

1. Get waypoints in ego frame from the action dictionary
2. Flip y-axis: `waypoints * [1, -1]` (CARLA convention: positive y = left)
3. Project to pixel space, divide by 2 (for half-res images 800×450)
4. Create blank canvas same size as the RGB image
5. Draw filled circles at projected locations:
   - Waypoints: radius=20 (colored cyan/red alternating)
   - Path points: radius=10 (colored green)
6. Store as the mask target

### Loss Computation

**Source**: `carformer/carformer/ponderer.py:484-494`

ETA's full pipeline:
1. The mask is processed through the vision model's patch embedding layer
2. Patch embeddings are reduced via `.abs().sum(1)` to get per-patch scalar labels
3. A BEV decoder (cross-attention between BEV tokens and action tokens) predicts mask logits
4. **BCE with logits loss** between predicted logits and binary labels (thresholded at 0.5)
5. Loss weight: **0.0625** (from `config/training/quantized.yaml:13`)

```python
# ponderer.py:484-494
vision_patch_labels = (mask > 0.5).float()
image_loss = F.binary_cross_entropy_with_logits(mask_logits, vision_patch_labels)
total_loss += 0.0625 * image_loss
```

### Simplified Approach for CIL++

Since CIL++ uses a ResNet+Transformer (not a ViT with patch embeddings), the adaptation is:
1. Project waypoints using the same math (already computed in `__getitem__`)
2. Create a **low-resolution binary mask** (e.g., 18×10 matching ~16:9 aspect ratio)
3. Add a 2-layer MLP mask head from the transformer's pooled output
4. Apply BCE with logits loss, weight 0.0625

---

## 2. Data Bucketing and Weighted Sampling

### Bucket Definitions

**Source**: `carformer/carformer/data/data_parser.py:328-474` — `get_all_buckets()`

16 buckets per sample, forming a binary vector. Each sample belongs to multiple buckets:

| # | Name | Condition | Threshold Values |
|---|------|-----------|-----------------|
| 0 | general | Always 1 | — |
| 1 | acc_scratch | throttle > 0.2 AND brake < 1.0 AND speed < 0.05 | Starting from standstill |
| 2 | acc_light_pedal | 0.2 < throttle < 0.5 | Light acceleration |
| 3 | acc_medium_pedal | 0.5 < throttle < 0.9 | Medium acceleration |
| 4 | acc_heavy_pedal | throttle > 0.9 | Full throttle |
| 5 | acc_brake | brake > 0.2 | Active braking |
| 6 | acc_coast | throttle < 0.2 AND brake < 1.0 | No input |
| 7 | steer_right | steer > 0.2 | Right turn |
| 8 | steer_left | steer < -0.2 | Left turn |
| 9 | vehicle_hazard_front | vehicle within front 30° cone | Uses `get_hazard_directions()` |
| 10 | vehicle_hazard_back | vehicle within rear 30° cone (>150°) | Uses `get_hazard_directions()` |
| 11 | vehicle_hazard_side | vehicle at 30°-150° | Uses `get_hazard_directions()` |
| 12 | stop_sign | traffic.stop affecting ego | `affects_ego == True` |
| 13 | red_light | red traffic light affecting ego | `state == 0` and `affects_ego` |
| 14 | swerving | swerving scenario + abs(steer) > 0.1 | Route name matching |
| 15 | pedestrian | walker collision hazard | Uses `is_walker_hazard()` |

### Hazard Detection Helpers

**Source**: `carformer/carformer/data/data_utils.py:388-503`

#### `get_hazard_directions(vehicle_list)` (line 388)
- Finds ego vehicle from bounding_boxes (`class == "ego_vehicle"`)
- For each other vehicle (`base_type == "car"`):
  - Computes distance and angle from ego's orientation
  - Filters by: `angle_to_car < 30°` AND `distance < max(2, 3 * ego_speed)`
  - Also checks: heading difference > 60° only if not very close
- Returns list of `angle_from_ego` values for hazardous vehicles

#### `is_walker_hazard(objects_list)` (line 474)
- Finds all walkers from bounding_boxes (`class == "walker"`)
- For each walker, predicts collision using `get_collision()`:
  - Ego trajectory: position + 10× orientation vector
  - Walker trajectory: position - 3× velocity + 8× velocity
  - Collision if both time-to-intersection values are in [0, 4] seconds
- Returns True if any collision predicted

#### `_orientation(yaw)` (line 457)
```python
def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])
```

### Swerving Scenarios (line 345-354)
The swerving bucket is only set if the route folder path contains one of:
```
"Accident", "BlockedIntersection", "ConstructionObstacle",
"HazardAtSideLane", "ParkedObstacle", "VehicleOpensDoorTwoWays"
```
AND `abs(steer) > 0.1`.

### Weight Computation Strategies

**Source**: `carformer/carformer/ponderer_lit.py:329-365`

Given: `weights` matrix of shape `(N_samples, 16)` with binary bucket memberships.

#### Strategy 1: "uniform" (line 342-345)
```python
weights = np.asarray(initial_weights)      # (N, 16)
weights = (weights / weights.sum(0))       # normalize each column by its total
weights = weights.mean(-1)                 # average across all 16 buckets → (N,)
```
**Effect**: Each bucket contributes equal total mass. Rare buckets (e.g., swerving)
get dramatically higher per-sample weights.

#### Strategy 2: "preferturns" (line 346-351)
```python
weights = np.asarray(initial_weights)      # (N, 16)
weights = weights / weights.sum(0)         # normalize columns
bucket_w = np.asarray(config.weights)      # (16,) user-specified multipliers
weights = (weights * bucket_w).sum(-1) / bucket_w.sum()  # weighted average → (N,)
```
**Effect**: Explicitly boosts certain behaviors. Default config gives 3× to steering, 2× to moderate acceleration.

### Configured Bucket Weights

**Source**: `carformer/carformer/config/training/default.yaml:53-72`

```yaml
bucket_weights:
  type: uniform
  total_ratio: 0.6       # Use 60% of dataset per epoch
  weights:
    - 1.0  # general
    - 1.0  # acc_scratch
    - 2.0  # acc_light_pedal         ← 2× emphasis
    - 2.0  # acc_medium_pedal        ← 2× emphasis
    - 1.0  # acc_heavy_pedal
    - 1.0  # acc_brake
    - 1.0  # acc_coast
    - 3.0  # steer_right             ← 3× emphasis
    - 3.0  # steer_left              ← 3× emphasis
    - 1.0  # vehicle_hazard_front
    - 1.0  # vehicle_hazard_back
    - 1.0  # vehicle_hazard_side
    - 1.0  # stop_sign
    - 1.0  # red_light
    - 1.0  # swerving
    - 1.0  # pedestrian
```

### Sampling Implementation

**Source**: `carformer/carformer/utils/distributedsampler.py`

`WeightedDistributedSampler` key behavior:
- `num_samples = ceil(len(dataset) * subsample_ratio / num_replicas)`
- Per epoch: `torch.multinomial(weights, total_size, replacement=True, generator=seeded_rng)`
- Epoch-seeded RNG ensures deterministic reproducibility
- Multi-GPU: interleaved rank splitting (`indices[rank::num_replicas]`)
- Tracks `last_indices` for distribution logging

---

## 3. Integration Plan for CIL++

See `/root/.claude/plans/serialized-petting-sloth.md` for the full implementation plan.

### Summary of Changes Needed

| File | Changes |
|------|---------|
| `cilpp/config.py` | Add mask loss + weighted sampling config fields |
| `cilpp/data.py` | Add waypoint projection, mask generation, bucket loading, `get_sample_weights()` |
| `cilpp/model.py` | Add mask prediction head, update forward returns |
| `train.py` | Add mask loss to training step, add weighted sampler, new CLI args |
| `gen_b2d_data.py` | Extract bounding_boxes, port hazard helpers, compute 16 buckets, store in .npy |
| `cilpp/weighted_sampler.py` (new) | Port ETA's `WeightedDistributedSampler` for DDP |

### Key Reference Files in ETA

| File | What to Port |
|------|-------------|
| `carformer/carformer/visualization/visutils.py:167-195` | `point_to_canvas_coordinates_rel_to_center()` projection math |
| `carformer/carformer/data/data_parser.py:328-474` | `get_all_buckets()` bucket definitions and thresholds |
| `carformer/carformer/data/data_utils.py:388-503` | `get_hazard_directions()`, `is_walker_hazard()`, `_orientation()`, `get_collision()` |
| `carformer/carformer/ponderer_lit.py:329-365` | Weight computation (uniform/preferturns strategies) |
| `carformer/carformer/utils/distributedsampler.py:8-153` | `WeightedDistributedSampler` with multinomial sampling |
| `carformer/carformer/ponderer.py:484-494` | Mask loss computation (BCE with logits, weight 0.0625) |
| `carformer/carformer/config/training/default.yaml:52-72` | Bucket weight config with all values |
