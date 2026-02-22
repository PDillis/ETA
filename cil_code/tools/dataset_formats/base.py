"""
Abstract base class for dataset format adapters.

This module defines the interface that all dataset format adapters must implement
to enable configurable dataset creation with different directory structures and
annotation formats.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional


class DatasetFormatAdapter(ABC):
    """
    Abstract base class for dataset-specific format adapters.

    Each dataset format (Bench2Drive, Play2Drive, etc.) should implement this
    interface to enable the generic dataset creation pipeline to work with
    different directory structures, annotation formats, and action encodings.
    """

    @abstractmethod
    def get_annotation_path(self, route_folder: str, frame_idx: int) -> str:
        """
        Return the path to the annotation file for a given frame.

        Args:
            route_folder: Path to the route directory
            frame_idx: Frame index (0-based)

        Returns:
            Full path to the annotation file

        Example:
            Bench2Drive: /path/to/route/anno/00000.json.gz
            Play2Drive: /path/to/route/cmd_fix_can_bus000000.json
        """
        pass

    @abstractmethod
    def load_annotation(self, path: str) -> Dict:
        """
        Load and parse an annotation file.

        Args:
            path: Path to the annotation file

        Returns:
            Dictionary containing annotation data with at least these keys:
            - 'action': int or tuple (action index or raw action values)
            - 'command': int (navigation command)
            - 'speed': float (vehicle speed in m/s)
            - Additional keys are dataset-specific

        Example:
            Bench2Drive: gzip.open() -> json.load()
            Play2Drive: open() -> json.load()
        """
        pass

    @abstractmethod
    def get_image_path(self, route_folder: str, camera: str, frame_idx: int) -> str:
        """
        Return the path to an image file for a given camera and frame.

        Args:
            route_folder: Path to the route directory
            camera: Camera name (e.g., 'rgb_front', 'rgb_central')
            frame_idx: Frame index (0-based)

        Returns:
            Full path to the image file

        Example:
            Bench2Drive: /path/to/route/camera/rgb_front/00000.jpg
            Play2Drive: /path/to/route/rgb_central000000.jpg
        """
        pass

    @abstractmethod
    def get_action(self, action_index: int) -> Tuple[float, float, float, float]:
        """
        Decode action index to raw action values.

        Args:
            action_index: Discrete action index

        Returns:
            Tuple of (throttle, steer, brake, hand_brake)

        Example:
            Bench2Drive: 39 discrete actions from Discrete_Actions_DICT
            Play2Drive: Custom action encoding (to be defined)
        """
        pass

    @abstractmethod
    def list_routes(self, dataset_root: str, split: str, val_routes: Optional[List[str]] = None) -> List[str]:
        """
        Return list of route directories to process for the given split.

        Args:
            dataset_root: Root directory of the dataset
            split: 'train' or 'val'
            val_routes: Optional list of validation route names (for filtering)

        Returns:
            List of full paths to route directories

        Example:
            Bench2Drive: Filter by val_list if split == 'val'
            Play2Drive: Scan {weather}/{weather}_route{N:05d}/ directories
        """
        pass

    @abstractmethod
    def get_camera_names(self) -> List[str]:
        """
        Return the list of available camera names for this dataset.

        Returns:
            List of camera names (e.g., ['rgb_front_left', 'rgb_front', 'rgb_front_right'])
        """
        pass

    @abstractmethod
    def get_reference_camera(self) -> str:
        """
        Return the reference camera name (whose path is stored in .npy).

        Returns:
            Reference camera name (e.g., 'rgb_front')
        """
        pass

    def get_route_group(self, route_name: str) -> str:
        """
        Return the scenario group for a route name, used for progress display grouping.

        Default: returns the full route name (one bar per route = no grouping).
        Override in dataset-specific adapters to group related routes together.

        Args:
            route_name: Directory name of the route (basename only, not full path)

        Returns:
            Group label string (e.g. scenario type, weather condition, etc.)

        Example overrides:
            Bench2Drive: 'HardBreakRoute_Town01_Route32_Weather6' → 'HardBreakRoute'
            Play2Drive:  'ClearNoon_route00001' → 'ClearNoon'
        """
        return route_name

    def get_sensor_path(self, route_folder: str, sensor_name: str, frame_idx: int) -> str:
        """
        Return the path to a sensor file (any type: rgb, depth, semantic, lidar, etc.).

        Args:
            route_folder: Path to the route directory
            sensor_name: Full sensor name (e.g., 'rgb_front', 'depth_front_left', 'lidar_top')
            frame_idx: Frame index (0-based)

        Returns:
            Full path to the sensor file
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement get_sensor_path()")

    def validate_route(self, route_folder: str) -> bool:
        """
        Optional: Validate that a route directory has the expected structure.

        Args:
            route_folder: Path to the route directory

        Returns:
            True if valid, False otherwise

        Default implementation always returns True.
        """
        return True
