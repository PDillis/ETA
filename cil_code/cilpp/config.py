import os

class GlobalConfig:
	"""Unified configuration (no YAML merge needed)."""

	#################
	#### CIL ++ #####
	#################

	targets = ['throttle', 'steer', 'brake']

	# ==== Data / sequence ====
	seq_len   = 1   # ENCODER_INPUT_FRAMES_NUM
	pred_len  = 8   # DECODER_OUTPUT_FRAMES_NUM  (set to 8 if training 8wp end-to-end)

	# Root directory for assets (images are referenced inside the npy if needed)
	# Set via environment variable: BENCH2DRIVE_ROOT
	root_dir_all = os.getenv("BENCH2DRIVE_ROOT")

	# NPY packs produced by your TCP packer (filtered 8wp set)
	# Set via environment variables: TRAIN_DATA_PATH, VAL_DATA_PATH
	train_data = os.getenv("TRAIN_DATA_PATH")
	val_data = os.getenv("VAL_DATA_PATH")

	# Multi-view camera usage (names must match dataset directory names)
	data_used = ["rgb_front_left", "rgb_front", "rgb_front_right"]
	reference_camera = "rgb_front"  # camera whose path is stored in the .npy; other paths are derived from it

	# ---- Image shape ----
	image_shape = [3, 300, 300]   # [C, H, W]

	# Normalization (ImageNet defaults; dataset-dependent)
	img_mean = [0.485, 0.456, 0.406]
	img_std  = [0.229, 0.224, 0.225]
	speed_max = 12.0  # max speed in dataset (m/s), used for normalization
	data_command_one_hot = True
	data_command_class_num = 6
	data_normalization = {
		"speed": [-3.3550484514579453, 17.265952633313525],
		# "speed": [-1.0, 11.0]
	}

	# ==== Training ====
	data_parallel = False
	batch_size = 64
	num_workers = 16
	num_epochs = 100

	loss_weights = {
		"throttle": 0.25,
		"steer":    0.50,
		"brake":    0.25,
	}

	# CIL++ Optimizer / LR schedule
	lr = 1e-4                 # LEARNING_RATE
	lr_min = 1e-5             # LEARNING_RATE_MINIMUM
	lr_milestones = [30, 50, 65]  # LEARNING_RATE_DECAY_EPOCHES
	lr_decay_level = 0.5          # LEARNING_RATE_POLICY.level

	# Loss weights used in your trainer
	speed_weight    = 0.05
	value_weight    = 0.001
	features_weight = 0.05

	# Augmentation
	img_aug = False  # from _g_conf.AUGMENTATION

	# TCP Controller
	turn_KP = 0.75
	turn_KI = 0.75
	turn_KD = 0.3
	turn_n = 40 # buffer size

	speed_KP = 5.0
	speed_KI = 0.5
	speed_KD = 1.0
	speed_n = 40 # buffer size

	max_throttle = 0.75 # upper limit on throttle signal value in dataset
	brake_speed = 0.4 # desired speed below which brake is triggered
	brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
	clip_delta = 0.25 # maximum change in speed input to logitudinal controller


	aim_dist = 4.0 # distance to search around for aim point
	angle_thresh = 0.3 # outlier control detection angle
	dist_thresh = 10 # target point y-distance for outlier filtering

	speed_weight = 0.05
	value_weight = 0.001
	features_weight = 0.05

	img_aug = False

	# ==== Mask Loss ====
	mask_loss_enabled = False
	mask_loss_weight = 0.0625           # from ETA's loss_params.mask_loss
	mask_height = 10                    # low-res mask rows
	mask_width = 18                     # low-res mask cols (~16:9)
	mask_waypoint_radius = 1            # radius in low-res mask pixels
	# Bench2Drive front camera intrinsics (hardcoded for now)
	camera_focal_length = 1142.5184053936916
	camera_cx = 800.0
	camera_cy = 450.0
	camera_height = -1.6                # camera height relative to ground
	camera_original_size = (1600, 900)  # original image resolution

	# ==== Weighted Sampling ====
	weighted_sampling = False
	bucket_weight_type = 'uniform'      # 'uniform' or 'preferturns'
	subsample_ratio = 1.0               # 1.0 = full dataset, 0.6 = 60%
	bucket_weights = [
		1.0,  # general
		1.0,  # acc_scratch
		2.0,  # acc_light_pedal
		2.0,  # acc_medium_pedal
		1.0,  # acc_heavy_pedal
		1.0,  # acc_brake
		1.0,  # acc_coast
		3.0,  # steer_right
		3.0,  # steer_left
		1.0,  # vehicle_hazard_front
		1.0,  # vehicle_hazard_back
		1.0,  # vehicle_hazard_side
		1.0,  # stop_sign
		1.0,  # red_light
		1.0,  # swerving
		1.0,  # pedestrian
	]

	# ==== Model ====
	backbone = "resnet34"  # backbone architecture (overridable via CLI)
	imagenet_pre_trained = True
	backbone_layer_id = 4  # which layer to extract features from (ResNet-specific)
	model_configuration = {
		"TxEncoder": {
			"d_model": 512,
			"n_head": 4,
			"num_layers": 4,
			"norm_first": True,
			"learnable_pe": True,
		},
		"command": {
			"fc": {
				"neurons": [512],
				"dropouts": [0.0],
			}
		},
		"speed": {
			"fc": {
				"neurons": [512],
				"dropouts": [0.0],
			}
		},
		"action_output": {
			"fc": {
				"neurons": [512, 256],
				"dropouts": [0.0, 0.0],
			}
		},
	}

	# Logging (trainer overrides can still be applied via CLI)
	# Set via environment variable: TRAINING_LOG_DIR
	log_dir = os.getenv("TRAINING_LOG_DIR")

	###########
	# ACTIONS
	###########

	# targets = ['throttle', 'steer', 'brake'] # From the float data, the ones that the network should estimate

	def __init__(self, **kwargs):
		# allow overriding via kwargs
		for k, v in kwargs.items():
			setattr(self, k, v)