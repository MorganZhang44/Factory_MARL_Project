"""Authoritative defaults for the environment rebuild line."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_USD = PROJECT_ROOT / "simulation" / "assets" / "scenes" / "slam_scene.usda"

DOGS = {
    "agent_1": {
        "prim_path": "/World/Actors/agent_1",
        "pos": (-2.0, -2.0, 0.42),
        "yaw_deg": 0.0,
    },
    "agent_2": {
        "prim_path": "/World/Actors/agent_2",
        "pos": (-2.0, 1.6, 0.42),
        "yaw_deg": 0.0,
    },
}

INTRUDER = {
    "prim_path": "/World/Actors/intruder_1",
    "pos": (2.0, -0.5, 1.34),
    "yaw_deg": 180.0,
}

INTRUDER_ROUTE_ANCHORS = (
    (2.0, -0.5, 1.34),
    (2.0, 2.3, 1.34),
    (-1.0, 2.3, 1.34),
    (-3.8, 2.3, 1.34),
    (-3.8, 0.1, 1.34),
    (-3.2, -2.6, 1.34),
    (0.2, -3.1, 1.34),
    (2.8, -2.2, 1.34),
    (2.0, -0.5, 1.34),
)

INTRUDER_ROUTE_STEP_SIZE = 0.05
INTRUDER_SPEED_MPS = 0.4

INTRUDER_ARM_DOWN_POSE = {
    "left_upper_arm:0": 0.9,
    "left_upper_arm:2": -0.8,
    "right_upper_arm:0": 0.9,
    "right_upper_arm:2": -0.8,
    "left_lower_arm": -2.0,
    "right_lower_arm": -2.0,
}

FLOOR_PRIM_PATH = "/World/SlamScene/Map/Floor"
REFERENCED_PHYSICS_SCENE_PATH = "/World/SlamScene/PhysicsScene"

FLOOR_PHYSICS_MATERIAL = {
    "prim_path": "/World/PhysicsMaterials/RebuildFloor",
    "static_friction": 1.0,
    "dynamic_friction": 1.0,
    "restitution": 0.0,
    "friction_combine_mode": "multiply",
    "restitution_combine_mode": "multiply",
}

ROBOT_PHYSICS_MATERIAL = {
    "prim_path": "/World/PhysicsMaterials/RebuildGo2Body",
    "static_friction": 0.8,
    "dynamic_friction": 0.6,
    "restitution": 0.0,
}

CCTV_CAMERA_NAMES = ("cam_nw", "cam_ne", "cam_e_upper", "cam_e_lower", "cam_se", "cam_sw")
CCTV_PITCH_DEG = 25.0
CCTV_HEIGHT = 2.35

DOG_CAMERA_USD_ROT_WXYZ = (0.5, 0.5, -0.5, -0.5)
DOG_CAMERA_POS = (0.3, 0.0, 0.1)
DOG_CAMERA_FOCAL_LENGTH = 3.5
DOG_CAMERA_HORIZONTAL_APERTURE = 12.0

CCTV_CAMERA_FOCAL_LENGTH = 14.0
CCTV_CAMERA_HORIZONTAL_APERTURE = 20.955

PERCEPTION_LIDAR_MOUNT_POS = (0.0, 0.0, 0.35)
PERCEPTION_LIDAR_MAX_DISTANCE = 50.0
PERCEPTION_LIDAR_VERTICAL_FOV_DEG = (-45.0, 45.0)
PERCEPTION_LIDAR_HORIZONTAL_FOV_DEG = (-180.0, 180.0)

LOCALIZATION_MESH_PATH = "/World/LocalizationStaticMesh"
