# Copyright (c) 2026, Multi-Agent Surveillance Project
# Sensor configuration constants for surveillance cameras, dog cameras, and LiDAR.
#
# References:
#   - Unitree Go2 front camera: 120° FOV, 1080p@15fps
#   - Unitree Go2 4D LiDAR: 360°×90° FOV, 21600 pts/sec
#   - Velodyne VLP-16 as LiDAR baseline config

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns


# ---------------------------------------------------------------------------
# Infrared post-processing config (pseudo-IR from depth + grayscale)
# ---------------------------------------------------------------------------
IR_POSTPROCESS_CFG = {
    "enabled": True,
    "colormap": "inferno",       # matplotlib colormap for heat-map style
    "depth_min": 0.5,            # meters – clip near
    "depth_max": 30.0,           # meters – clip far
    "blend_alpha": 0.6,          # blend ratio between depth-heatmap and grayscale RGB
}


# ---------------------------------------------------------------------------
# Surveillance (fixed / CCTV) Camera Configuration
# ---------------------------------------------------------------------------
SURVEILLANCE_CAM_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/SurveillanceCam",          # overridden per instance
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=12.0,          # mm – moderate FOV (~60°)
        horizontal_aperture=20.955, # mm – standard 35mm-equivalent
    ),
    width=320,
    height=240,
    data_types=["rgb", "semantic_segmentation"],
    colorize_semantic_segmentation=False,   # raw int32 IDs for detection logic
    update_period=0.1,                      # 10 Hz update rate
    offset=CameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        convention="world",
    ),
)
"""Fixed surveillance camera. 60° FOV, 640×480, RGB+semantic seg. (No Depth)"""


# ---------------------------------------------------------------------------
# Dog-Mounted (Unitree Go2) Camera Configuration
# 120° FOV to match real Go2 front wide-angle camera
# ---------------------------------------------------------------------------
DOG_CAM_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/DogCam",                   # overridden per instance
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=3.5,            # mm – short focal length for wide FOV
        horizontal_aperture=12.0,    # mm – yields ~120° horizontal FOV
        # hfov ≈ 2 * atan(aperture / (2 * focal_length))
        #      ≈ 2 * atan(12 / 7) ≈ 2 * 59.7° ≈ 119.4°
    ),
    width=320,
    height=240,
    data_types=["rgb", "depth", "semantic_segmentation"],
    colorize_semantic_segmentation=False,
    update_period=0.067,   # ~15 Hz matching Go2 camera spec
    offset=CameraCfg.OffsetCfg(
        pos=(0.3, 0.0, 0.1),        # front of dog, slightly above base
        rot=(1.0, 0.0, 0.0, 0.0),
        convention="world",
    ),
)
"""Dog-mounted wide-angle camera. 120° FOV, 640×480, RGB+depth+semantic seg."""


# ---------------------------------------------------------------------------
# Dog-Mounted LiDAR Configuration
# Matches Unitree Go2 4D LiDAR: 360°×90° hemispherical FOV
# Using RayCaster with LidarPattern (similar to Velodyne VLP-16 but wider vertical)
# ---------------------------------------------------------------------------
DOG_LIDAR_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/DogLidar",                 # overridden per instance
    ray_alignment="base",
    pattern_cfg=patterns.LidarPatternCfg(
        channels=16,
        vertical_fov_range=(-45.0, 45.0),      # 90° total vertical FOV
        horizontal_fov_range=(-180.0, 180.0),   # 360° horizontal FOV
        horizontal_res=1.0,                      # 1° resolution (360 rays per channel)
    ),
    offset=RayCasterCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.35),  # top of the dog body
    ),
    debug_vis=False,
    max_distance=50.0,          # 50m max range
    mesh_prim_paths=["/World/ground", "/World/envs"],  # include all scene objects
)
"""Dog-mounted LiDAR. 360°×90° FOV, 16 channels, 50m range."""


# ---------------------------------------------------------------------------
# Dog-Mounted IMU Configuration
# ---------------------------------------------------------------------------
from isaaclab.sensors import ImuCfg

DOG_IMU_CFG = ImuCfg(
    prim_path="{ENV_REGEX_NS}/DogBase",                  # attached to base link
    update_period=0.01,                                  # 100 Hz high frequency
    offset=ImuCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    gravity_bias=(0.0, 0.0, 9.81),                       # internal gravity compensation
)


# ---------------------------------------------------------------------------
# Helper: compute camera intrinsic matrix from config
# ---------------------------------------------------------------------------
def get_intrinsics_from_camera_cfg(cam_cfg: CameraCfg) -> tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy) from a CameraCfg.

    Uses pinhole model:
        fx = focal_length * width / horizontal_aperture
        fy = fx (square pixels assumed)
        cx = width / 2
        cy = height / 2
    """
    spawn_cfg = cam_cfg.spawn
    f_mm = spawn_cfg.focal_length
    aperture_mm = spawn_cfg.horizontal_aperture
    w = cam_cfg.width
    h = cam_cfg.height

    fx = f_mm * w / aperture_mm
    fy = fx  # square pixels
    cx = w / 2.0
    cy = h / 2.0
    return (fx, fy, cx, cy)
