"""
Play2Drive dataset format adapter (in-house data collection).

This adapter implements the Play2Drive-specific directory structure:
- Directory pattern: {weather}/{weather}_route{route_number:05d}/
- Annotations: route_folder/cmd_fix_can_bus{frame:06d}.json (JSON, not gzip)
- Images: route_folder/{camera}{frame:06d}.jpg (e.g., rgb_central000000.jpg)
- Action encoding: To be defined based on data collection setup

NOTE: This is a stub implementation. The exact annotation format, action encoding,
and camera names need to be finalized based on the actual data collection pipeline.
"""

import os
import json
import glob
from typing import List, Dict, Tuple, Optional

from .base import DatasetFormatAdapter


class Play2DriveAdapter(DatasetFormatAdapter):
    """
    Adapter for the Play2Drive in-house data collection format.

    Directory structure:
        dataset_root/
            {weather}/
                {weather}_route00000/
                    cmd_fix_can_bus000000.json
                    rgb_central000000.jpg
                    rgb_left000000.jpg
                    rgb_right000000.jpg
                    ...
                {weather}_route00001/
                    ...

    NOTE: This is a stub implementation. Adjust based on actual data format.
    """

    def __init__(self):
        """Initialize the Play2Drive adapter."""
        pass

    def get_annotation_path(self, route_folder: str, frame_idx: int) -> str:
        """
        Return path to JSON annotation file.

        NOTE: Adjust naming pattern based on actual data format.
        """
        return os.path.join(route_folder, f'cmd_fix_can_bus{frame_idx:06d}.json')

    def load_annotation(self, path: str) -> Dict:
        """
        Load and parse JSON annotation file.

        NOTE: Adjust based on actual annotation schema.
        Expected keys: 'action', 'command', 'speed', etc.
        """
        with open(path, 'r') as f:
            return json.load(f)

    def get_image_path(self, route_folder: str, camera: str, frame_idx: int) -> str:
        """
        Return path to image file.

        NOTE: Adjust camera naming convention based on actual data format.
        Example: rgb_central000000.jpg, rgb_left000000.jpg, etc.
        """
        return os.path.join(route_folder, f'{camera}{frame_idx:06d}.jpg')

    def get_action(self, action_index: int) -> Tuple[float, float, float, float]:
        """
        Decode action index to raw action values.

        NOTE: This needs to be implemented based on the actual action encoding
        used in the Play2Drive data collection pipeline.

        Args:
            action_index: Discrete action index

        Returns:
            Tuple of (throttle, steer, brake, hand_brake)

        Raises:
            NotImplementedError: This method must be implemented based on the
                                 actual action encoding scheme.
        """
        raise NotImplementedError(
            "Play2Drive action encoding not yet defined. "
            "Implement this method based on the data collection action space."
        )

    def list_routes(self, dataset_root: str, split: str, val_routes: Optional[List[str]] = None) -> List[str]:
        """
        Return list of route directories for train/val split.

        Scans {weather}/{weather}_route{N:05d}/ directories.

        NOTE: Implement train/val split logic based on your requirements.
        Options:
        1. Provide a val_routes file with explicit route names
        2. Use a percentage split (e.g., 80/20)
        3. Use weather-based split (certain weather conditions for val)

        Args:
            dataset_root: Root directory containing weather subdirectories
            split: 'train' or 'val'
            val_routes: Optional list of validation route names

        Returns:
            List of full paths to route directories

        Raises:
            NotImplementedError: Implement based on your split strategy
        """
        all_routes = []

        # Scan weather directories
        for weather_dir in os.listdir(dataset_root):
            weather_path = os.path.join(dataset_root, weather_dir)
            if not os.path.isdir(weather_path):
                continue

            # Scan route directories within each weather folder
            pattern = os.path.join(weather_path, f'{weather_dir}_route*')
            routes = glob.glob(pattern)
            all_routes.extend(routes)

        # TODO: Implement train/val split logic
        # Example: Use val_routes list if provided
        if val_routes is not None:
            val_set = set(os.path.basename(r) for r in val_routes)
            if split == 'train':
                return [r for r in all_routes if os.path.basename(r) not in val_set]
            elif split == 'val':
                return [r for r in all_routes if os.path.basename(r) in val_set]
        else:
            raise NotImplementedError(
                "Play2Drive train/val split not yet defined. "
                "Provide val_routes list or implement a split strategy."
            )

    def get_camera_names(self) -> List[str]:
        """
        Return list of available camera names in Play2Drive.

        NOTE: Adjust based on actual camera setup.
        """
        # TODO: Adjust based on actual camera names in data collection
        return ['rgb_left', 'rgb_central', 'rgb_right']

    def get_reference_camera(self) -> str:
        """
        Return the reference camera name (whose path is stored in .npy).

        NOTE: Adjust based on your preference.
        """
        # TODO: Choose the reference camera
        return 'rgb_central'

    def get_route_group(self, route_name: str) -> str:
        """
        Extract group from route name: 'ClearNoon_route00001' → 'ClearNoon'.

        NOTE: Adjust the split pattern based on actual Play2Drive naming convention.
        """
        return route_name.split('_route')[0] if '_route' in route_name else route_name

    def validate_route(self, route_folder: str) -> bool:
        """
        Validate that a route directory has the expected Play2Drive structure.

        NOTE: Implement validation based on actual data format.

        Args:
            route_folder: Path to the route directory

        Returns:
            True if valid, False otherwise
        """
        # Example: Check if at least one annotation file exists
        pattern = os.path.join(route_folder, 'cmd_fix_can_bus*.json')
        has_annotations = len(glob.glob(pattern)) > 0
        return has_annotations
