# Copyright (c) 2026, Multi-Agent Surveillance Project
# Scene configuration: imported SLAM scene with robot dogs, intruder, and sensors.

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import H1_MINIMAL_CFG, UNITREE_GO2_CFG


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SLAM_SCENE_USD = PROJECT_ROOT / "isaac" / "assets" / "scenes" / "slam_scene.usda"

# The standalone validator creates this same RTX LiDAR model with IsaacSensorCreateRtxLidar.
RPLIDAR_S2E_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/Sensors/Slamtec/RPLIDAR_S2E/Slamtec_RPLIDAR_S2E.usd"
)

AGENT_1_POS = (-2.0, -2.0, 0.42)
AGENT_2_POS = (-2.0, 1.6, 0.42)
INTRUDER_POS = (2.0, -0.5, 1.05)

DOG_JOINT_POS = {
    ".*L_hip_joint": 0.1,
    ".*R_hip_joint": -0.1,
    "F[L,R]_thigh_joint": 0.8,
    "R[L,R]_thigh_joint": 1.0,
    ".*_calf_joint": -1.5,
}

SEMANTIC_LABELS = {
    "class:intruder": (255, 50, 50, 255),
    "class:dog": (50, 150, 255, 255),
    "class:slam_scene": (150, 150, 150, 255),
}


def _go2_cfg(prim_path: str, pos: tuple[float, float, float]) -> ArticulationCfg:
    cfg = UNITREE_GO2_CFG.replace(prim_path=prim_path)
    cfg.init_state = ArticulationCfg.InitialStateCfg(
        pos=pos,
        joint_pos=DOG_JOINT_POS,
        joint_vel={".*": 0.0},
    )
    cfg.spawn.semantic_tags = [("class", "dog")]
    return cfg


def _h1_intruder_cfg(prim_path: str, pos: tuple[float, float, float]) -> ArticulationCfg:
    cfg = H1_MINIMAL_CFG.replace(prim_path=prim_path)
    cfg.init_state = ArticulationCfg.InitialStateCfg(
        pos=pos,
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    )
    cfg.spawn.semantic_tags = [("class", "intruder")]
    return cfg


def _front_camera_cfg(prim_path: str) -> CameraCfg:
    return CameraCfg(
        prim_path=prim_path,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 1000.0),
        ),
        width=640,
        height=480,
        data_types=["rgb", "depth", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        update_period=0.067,
        offset=CameraCfg.OffsetCfg(
            pos=(0.34, 0.0, 0.12),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )


def _raycast_lidar_cfg(prim_path: str, debug_vis: bool = False) -> RayCasterCfg:
    return RayCasterCfg(
        prim_path=prim_path,
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=16,
            vertical_fov_range=(-15.0, 15.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=0.2,
        ),
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.26, 0.0, 0.18),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        debug_vis=debug_vis,
        max_distance=50.0,
        mesh_prim_paths=["/World/SlamScene/Map/Floor"],
    )


def _rplidar_visual_cfg(prim_path: str) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=RPLIDAR_S2E_USD,
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.26, 0.0, 0.18),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


@configclass
class SlamSceneCfg(InteractiveSceneCfg):
    """Configuration for the imported SLAM scene.

    Layout:
    - Imported pure USDA scene at /World/SlamScene
    - Two Unitree Go2 robot dogs
    - One Unitree H1 intruder
    - One front camera and one LiDAR-style ray caster on each dog
    - Visible RPLIDAR_S2E model on each dog for viewport inspection
    """

    num_envs: int = 1
    env_spacing: float = 10.0
    replicate_physics: bool = False

    # ------------------------------------------------------------------
    # Environment: imported SLAM scene
    # ------------------------------------------------------------------
    slam_scene = AssetBaseCfg(
        prim_path="/World/SlamScene",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(SLAM_SCENE_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            semantic_tags=[("class", "slam_scene")],
        ),
    )

    # ------------------------------------------------------------------
    # Actors
    # ------------------------------------------------------------------
    agent_1: ArticulationCfg = _go2_cfg("{ENV_REGEX_NS}/agent_1", AGENT_1_POS)
    agent_2: ArticulationCfg = _go2_cfg("{ENV_REGEX_NS}/agent_2", AGENT_2_POS)
    intruder_1: ArticulationCfg = _h1_intruder_cfg("{ENV_REGEX_NS}/intruder_1", INTRUDER_POS)

    # ------------------------------------------------------------------
    # Dog cameras
    # ------------------------------------------------------------------
    agent_1_front_camera = _front_camera_cfg("{ENV_REGEX_NS}/agent_1/base/front_camera")
    agent_2_front_camera = _front_camera_cfg("{ENV_REGEX_NS}/agent_2/base/front_camera")

    # ------------------------------------------------------------------
    # Dog LiDARs
    # ------------------------------------------------------------------
    agent_1_lidar = _raycast_lidar_cfg("{ENV_REGEX_NS}/agent_1/base")
    agent_2_lidar = _raycast_lidar_cfg("{ENV_REGEX_NS}/agent_2/base")

    agent_1_lidar_model = _rplidar_visual_cfg("{ENV_REGEX_NS}/agent_1/base/front_lidar_model")
    agent_2_lidar_model = _rplidar_visual_cfg("{ENV_REGEX_NS}/agent_2/base/front_lidar_model")

    # ------------------------------------------------------------------
    # Lighting
    # ------------------------------------------------------------------
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=800.0,
            color=(0.9, 0.9, 1.0),
        ),
    )

    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=2500.0,
            color=(1.0, 0.95, 0.85),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=(0.866, 0.0, 0.5, 0.0),
        ),
    )
