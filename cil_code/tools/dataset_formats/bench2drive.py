"""
Bench2Drive dataset format adapter.

This adapter implements the Bench2Drive-specific directory structure:
- Annotations: route_folder/anno/{frame:05}.json.gz (gzip compressed JSON)
- Images: route_folder/camera/{camera}/{frame:05}.jpg
- Expert assessment: route_folder/expert_assessment/{frame:05}.npz
- 39 discrete actions from the Bench2Drive action space
"""

import os
import json
import gzip
from typing import List, Dict, Tuple, Optional

from .base import DatasetFormatAdapter


# Default validation routes from Bench2Drive paper (48 routes)
DEFAULT_VAL_ROUTES = [
    'StaticCutIn_Town05_Route226_Weather18',
    'MergerIntoSlowTrafficV2_Town12_Route857_Weather25',
    'YieldToEmergencyVehicle_Town04_Route166_Weather10',
    'ConstructionObstacle_Town10HD_Route74_Weather22',
    'VehicleTurningRoutePedestrian_Town15_Route445_Weather11',
    'VanillaSignalizedTurnEncounterRedLight_Town07_Route359_Weather21',
    'SignalizedJunctionLeftTurnEnterFlow_Town13_Route657_Weather2',
    'LaneChange_Town06_Route307_Weather21',
    'ConstructionObstacleTwoWays_Town12_Route1093_Weather1',
    'HazardAtSideLaneTwoWays_Town12_Route1151_Weather7',
    'OppositeVehicleTakingPriority_Town04_Route214_Weather6',
    'NonSignalizedJunctionRightTurn_Town03_Route126_Weather18',
    'VanillaNonSignalizedTurnEncounterStopsign_Town12_Route979_Weather9',
    'ParkedObstacle_Town06_Route282_Weather22',
    'ControlLoss_Town10HD_Route378_Weather14',
    'ControlLoss_Town04_Route170_Weather14',
    'OppositeVehicleRunningRedLight_Town04_Route180_Weather23',
    'InterurbanAdvancedActorFlow_Town06_Route324_Weather2',
    'HighwayCutIn_Town12_Route1029_Weather15',
    'MergerIntoSlowTraffic_Town06_Route317_Weather5',
    'NonSignalizedJunctionLeftTurn_Town07_Route342_Weather3',
    'AccidentTwoWays_Town12_Route1115_Weather23',
    'ParkingCrossingPedestrian_Town13_Route545_Weather25',
    'VanillaSignalizedTurnEncounterGreenLight_Town07_Route354_Weather8',
    'ParkingExit_Town12_Route922_Weather12',
    'VanillaSignalizedTurnEncounterRedLight_Town15_Route491_Weather23',
    'HardBreakRoute_Town01_Route32_Weather6',
    'DynamicObjectCrossing_Town01_Route3_Weather3',
    'ConstructionObstacle_Town12_Route78_Weather0',
    'EnterActorFlow_Town03_Route132_Weather2',
    'HazardAtSideLane_Town10HD_Route373_Weather9',
    'InvadingTurn_Town02_Route95_Weather9',
    'TJunction_Town05_Route260_Weather0',
    'VehicleTurningRoute_Town15_Route504_Weather10',
    'DynamicObjectCrossing_Town02_Route11_Weather11',
    'TJunction_Town06_Route306_Weather20',
    'ParkedObstacleTwoWays_Town13_Route1333_Weather26',
    'SignalizedJunctionRightTurn_Town03_Route118_Weather14',
    'NonSignalizedJunctionLeftTurnEnterFlow_Town12_Route949_Weather13',
    'VehicleOpensDoorTwoWays_Town12_Route1203_Weather7',
    'CrossingBicycleFlow_Town12_Route977_Weather15',
    'SignalizedJunctionLeftTurn_Town04_Route173_Weather26',
    'HighwayExit_Town06_Route312_Weather0',
    'Accident_Town05_Route218_Weather10',
    'ParkedObstacle_Town10HD_Route372_Weather8',
    'InterurbanActorFlow_Town12_Route1291_Weather1',
    'ParkingCutIn_Town13_Route1343_Weather1',
    'VehicleTurningRoutePedestrian_Town15_Route481_Weather19',
    'PedestrianCrossing_Town13_Route747_Weather19',
    'BlockedIntersection_Town03_Route135_Weather5',
]


# Bench2Drive discrete action space (39 actions)
DISCRETE_ACTIONS_DICT = {
    0:  (0, 0, 1, False),
    1:  (0.7, -0.5, 0, False),
    2:  (0.7, -0.3, 0, False),
    3:  (0.7, -0.2, 0, False),
    4:  (0.7, -0.1, 0, False),
    5:  (0.7, 0, 0, False),
    6:  (0.7, 0.1, 0, False),
    7:  (0.7, 0.2, 0, False),
    8:  (0.7, 0.3, 0, False),
    9:  (0.7, 0.5, 0, False),
    10: (0.3, -0.7, 0, False),
    11: (0.3, -0.5, 0, False),
    12: (0.3, -0.3, 0, False),
    13: (0.3, -0.2, 0, False),
    14: (0.3, -0.1, 0, False),
    15: (0.3, 0, 0, False),
    16: (0.3, 0.1, 0, False),
    17: (0.3, 0.2, 0, False),
    18: (0.3, 0.3, 0, False),
    19: (0.3, 0.5, 0, False),
    20: (0.3, 0.7, 0, False),
    21: (0, -1, 0, False),
    22: (0, -0.6, 0, False),
    23: (0, -0.3, 0, False),
    24: (0, -0.1, 0, False),
    25: (1, 0, 0, False),
    26: (0, 0.1, 0, False),
    27: (0, 0.3, 0, False),
    28: (0, 0.6, 0, False),
    29: (0, 1.0, 0, False),
    30: (0.5, -0.5, 0, True),
    31: (0.5, -0.3, 0, True),
    32: (0.5, -0.2, 0, True),
    33: (0.5, -0.1, 0, True),
    34: (0.5, 0, 0, True),
    35: (0.5, 0.1, 0, True),
    36: (0.5, 0.2, 0, True),
    37: (0.5, 0.3, 0, True),
    38: (0.5, 0.5, 0, True),
}


class Bench2DriveAdapter(DatasetFormatAdapter):
    """
    Adapter for the Bench2Drive dataset format.

    Directory structure:
        route_folder/
            anno/{frame:05}.json.gz
            camera/rgb_front/{frame:05}.jpg
            camera/rgb_front_left/{frame:05}.jpg
            camera/rgb_front_right/{frame:05}.jpg
            expert_assessment/{frame:05}.npz
    """

    def __init__(self, val_routes: Optional[List[str]] = None):
        """
        Initialize the Bench2Drive adapter.

        Args:
            val_routes: Optional list of validation route names.
                       If None, uses default validation routes from the paper.
        """
        self.val_routes = val_routes if val_routes is not None else DEFAULT_VAL_ROUTES

    def get_annotation_path(self, route_folder: str, frame_idx: int) -> str:
        """Return path to gzip-compressed JSON annotation file."""
        return os.path.join(route_folder, f'anno/{frame_idx:05}.json.gz')

    def load_annotation(self, path: str) -> Dict:
        """Load and parse gzip-compressed JSON annotation file."""
        with gzip.open(path, 'rt', encoding='utf-8') as gz_file:
            return json.load(gz_file)

    def get_image_path(self, route_folder: str, camera: str, frame_idx: int) -> str:
        """Return path to image file in camera/rgb_front/ structure."""
        return os.path.join(route_folder, f'camera/{camera}/{frame_idx:05}.jpg')

    def get_expert_assessment_path(self, route_folder: str, frame_idx: int) -> str:
        """
        Return path to expert assessment .npz file.

        This is Bench2Drive-specific and not part of the base adapter interface.
        """
        return os.path.join(route_folder, f'expert_assessment/{frame_idx:05}.npz')

    def get_action(self, action_index: int) -> Tuple[float, float, float, float]:
        """
        Decode Bench2Drive discrete action index to raw action values.

        Args:
            action_index: Integer in [0, 38]

        Returns:
            Tuple of (throttle, steer, brake, hand_brake)
        """
        if action_index not in DISCRETE_ACTIONS_DICT:
            raise ValueError(f"Invalid action index {action_index}. Must be in [0, 38].")
        return DISCRETE_ACTIONS_DICT[action_index]

    def list_routes(self, dataset_root: str, split: str, val_routes: Optional[List[str]] = None) -> List[str]:
        """
        Return list of route directories for train/val split.

        Args:
            dataset_root: Root directory containing all routes
            split: 'train' or 'val'
            val_routes: Optional override for validation routes (uses self.val_routes if None)

        Returns:
            List of full paths to route directories
        """
        if val_routes is None:
            val_routes = self.val_routes

        val_set = set(val_routes)
        routes = []

        for route_name in os.listdir(dataset_root):
            route_path = os.path.join(dataset_root, route_name)
            if not os.path.isdir(route_path):
                continue

            if split == 'train':
                if route_name not in val_set:
                    routes.append(route_path)
            elif split == 'val':
                if route_name in val_set:
                    routes.append(route_path)
            else:
                raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'val'.")

        return sorted(routes)

    def get_camera_names(self) -> List[str]:
        """Return list of available camera names in Bench2Drive."""
        return ['rgb_front_left', 'rgb_front', 'rgb_front_right']

    def get_reference_camera(self) -> str:
        """Return the reference camera name (whose path is stored in .npy)."""
        return 'rgb_front'

    def get_route_group(self, route_name: str) -> str:
        """Extract scenario type prefix: 'HardBreakRoute_Town01_Route32_Weather6' → 'HardBreakRoute'"""
        return route_name.split('_Town')[0]

    def validate_route(self, route_folder: str) -> bool:
        """
        Validate that a route directory has the expected Bench2Drive structure.

        Args:
            route_folder: Path to the route directory

        Returns:
            True if valid, False otherwise
        """
        required_subdirs = ['anno', 'camera', 'expert_assessment']
        for subdir in required_subdirs:
            if not os.path.isdir(os.path.join(route_folder, subdir)):
                return False
        return True
