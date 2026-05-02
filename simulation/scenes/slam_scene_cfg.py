# Copyright (c) 2026, Multi-Agent Surveillance Project
# Scene configuration converted from simulation/assets/scenes/slam_scene_with_actors.usda.

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ImuCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab_assets import HUMANOID_CFG
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


FLOOR_COLOR = (0.72, 0.72, 0.68)
WALL_COLOR = (0.08, 0.08, 0.08)
OBSTACLE_COLOR = (0.36, 0.14, 0.10)

AGENT_1_START_POS = (-2.0, -2.0, 0.42)
AGENT_2_START_POS = (-2.0, 1.6, 0.42)
INTRUDER_START_POS = (2.0, -0.5, 1.34)
CCTV_HEIGHT = 4.5
CCTV_PITCH_DEG = 25.0
CCTV_CORNER_SPECS = {
    "cam_nw": {"corner": (-5.50, 3.45), "look_hint": (-2.20, 1.90), "mount_inset": 1.05},
    "cam_ne": {"corner": (6.25, 3.45), "look_hint": (2.35, 0.85), "mount_inset": 1.10},
    "cam_e_upper": {"corner": (6.25, -0.95), "look_hint": (2.55, 0.55), "mount_inset": 0.95},
    "cam_e_lower": {"corner": (3.75, -0.95), "look_hint": (2.35, -0.65), "mount_inset": 0.85},
    "cam_se": {"corner": (3.75, -4.65), "look_hint": (2.10, -1.45), "mount_inset": 0.95},
    "cam_sw": {"corner": (-5.50, -4.65), "look_hint": (-2.30, -2.35), "mount_inset": 1.00},
}

DOG_JOINT_POS = {
    ".*L_hip_joint": 0.1,
    ".*R_hip_joint": -0.1,
    "F[L,R]_thigh_joint": 0.8,
    "R[L,R]_thigh_joint": 1.0,
    ".*_calf_joint": -1.5,
}

WALLS = (
    ("Wall_0000_Back", (9.52, 0.16, 1.8), (-0.85, 3.45, 0.9)),
    ("Wall_0001_Left", (0.16, 8.16, 1.8), (-5.5, -0.65, 0.9)),
    ("Wall_0002_Front", (9.18, 0.16, 1.8), (-0.95, -4.65, 0.9)),
    ("Wall_0003_RightLower", (0.16, 4.08, 1.8), (3.75, -2.85, 0.9)),
    ("Wall_0004_StepMiddle", (3.06, 0.16, 1.8), (5.0, -0.95, 0.9)),
    ("Wall_0005_RightUpper", (0.16, 4.93, 1.8), (6.25, 1.35, 0.9)),
    ("Wall_0006_TopRight", (2.89, 0.16, 1.8), (5.15, 3.45, 0.9)),
    ("Wall_0007_StepJoin", (0.16, 2.125, 1.8), (3.75, 2.65, 0.9)),
)

OBSTACLES = (
    ("Obstacle_0000_LeftIsland", (0.425, 1.275, 1.8), (-2.3, -0.85, 0.9)),
    ("Obstacle_0001_CenterIsland", (0.425, 1.375, 1.8), (0.4, -0.95, 0.9)),
    ("Obstacle_0002_RightIsland", (0.45, 1.4, 1.8), (2.95, -0.65, 0.9)),
)

RAYCAST_MESH_PRIM_PATHS = (
    "/World/LocalizationStaticMesh",
)


def _preview(color: tuple[float, float, float]) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.75)


def _static_cuboid_cfg(
    prim_path: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
    semantic_class: str,
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=_preview(color),
            semantic_tags=[("class", semantic_class)],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _go2_cfg(prim_path: str, pos: tuple[float, float, float]) -> ArticulationCfg:
    cfg = UNITREE_GO2_CFG.replace(prim_path=prim_path)
    cfg.init_state = ArticulationCfg.InitialStateCfg(
        pos=pos,
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=DOG_JOINT_POS,
        joint_vel={".*": 0.0},
    )
    cfg.spawn.semantic_tags = [("class", "dog")]
    return cfg


def _humanoid_intruder_cfg(prim_path: str, pos: tuple[float, float, float]) -> ArticulationCfg:
    cfg = HUMANOID_CFG.replace(prim_path=prim_path)
    cfg.init_state = ArticulationCfg.InitialStateCfg(
        pos=pos,
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    )
    cfg.spawn.semantic_tags = [("class", "suspect")]
    return cfg


def _front_camera_cfg(prim_path: str) -> CameraCfg:
    return CameraCfg(
        prim_path=prim_path,
        spawn=sim_utils.PinholeCameraCfg(
            clipping_range=(0.05, 1000.0),
            focal_length=3.5,
            focus_distance=400.0,
            horizontal_aperture=12.0,
        ),
        width=320,
        height=240,
        data_types=["rgb", "depth", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        update_period=0.067,
        offset=CameraCfg.OffsetCfg(
            pos=(0.3, 0.0, 0.1),
            rot=(0.5, 0.5, -0.5, -0.5),
            convention="world",
        ),
    )


def _perception_lidar_cfg(prim_path: str) -> RayCasterCfg:
    return RayCasterCfg(
        prim_path=prim_path,
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=16,
            vertical_fov_range=(-45.0, 45.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=1.0,
        ),
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.35),
        ),
        debug_vis=False,
        max_distance=50.0,
        mesh_prim_paths=list(RAYCAST_MESH_PRIM_PATHS),
    )


def _dog_imu_cfg(prim_path: str) -> ImuCfg:
    return ImuCfg(
        prim_path=prim_path,
        update_period=0.01,
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        gravity_bias=(0.0, 0.0, 9.81),
    )


def _move_toward(source_xy: tuple[float, float], target_xy: tuple[float, float], distance: float) -> tuple[float, float]:
    dx = target_xy[0] - source_xy[0]
    dy = target_xy[1] - source_xy[1]
    norm = math.hypot(dx, dy)
    if norm < 1.0e-6:
        return source_xy
    scale = distance / norm
    return (source_xy[0] + dx * scale, source_xy[1] + dy * scale)


def _cctv_cfg(name: str) -> CameraCfg:
    spec = CCTV_CORNER_SPECS[name]
    x, y = _move_toward(spec["corner"], spec["look_hint"], spec["mount_inset"])
    yaw = math.atan2(spec["look_hint"][1] - y, spec["look_hint"][0] - x)
    pitch = math.radians(CCTV_PITCH_DEG)
    quat = (
        math.cos(yaw * 0.5) * math.cos(pitch * 0.5),
        -math.sin(yaw * 0.5) * math.sin(pitch * 0.5),
        math.cos(yaw * 0.5) * math.sin(pitch * 0.5),
        math.sin(yaw * 0.5) * math.cos(pitch * 0.5),
    )
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=14.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 1000.0),
        ),
        width=640,
        height=480,
        data_types=["rgb", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        update_period=0.1,
        offset=CameraCfg.OffsetCfg(
            pos=(x, y, CCTV_HEIGHT),
            rot=quat,
            convention="world",
        ),
    )


def _cctv_marker(name: str) -> AssetBaseCfg:
    spec = CCTV_CORNER_SPECS[name]
    x, y = _move_toward(spec["corner"], spec["look_hint"], spec["mount_inset"])
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}_marker",
        spawn=sim_utils.CuboidCfg(
            size=(0.18, 0.18, 0.18),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.1, 0.1),
                emissive_color=(0.6, 0.0, 0.0),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, CCTV_HEIGHT + 0.18)),
    )


def _wall_cfg(name: str, size: tuple[float, float, float], pos: tuple[float, float, float]) -> AssetBaseCfg:
    return _static_cuboid_cfg(f"{{ENV_REGEX_NS}}/{name}", size, pos, WALL_COLOR, "wall")


def _obstacle_cfg(name: str, size: tuple[float, float, float], pos: tuple[float, float, float]) -> AssetBaseCfg:
    return _static_cuboid_cfg(f"{{ENV_REGEX_NS}}/{name}", size, pos, OBSTACLE_COLOR, "obstacle")


@configclass
class SlamSceneCfg(InteractiveSceneCfg):
    """SLAM scene expressed directly as Isaac Lab config.

    This is a programmatic conversion of ``slam_scene_with_actors.usda``:
    the static map comes from the referenced ``slam_scene.usda``, while the
    actors, front cameras, and perception-style RayCaster LiDARs are expressed
    directly with the current Simulation naming.
    """

    num_envs: int = 1
    env_spacing: float = 25.0
    replicate_physics: bool = False

    floor = _static_cuboid_cfg(
        "{ENV_REGEX_NS}/Floor",
        (16.0, 9.5, 0.02),
        (-0.85, -0.65, -0.02),
        FLOOR_COLOR,
        "floor",
    )

    wall_0000_back = _wall_cfg(*WALLS[0])
    wall_0001_left = _wall_cfg(*WALLS[1])
    wall_0002_front = _wall_cfg(*WALLS[2])
    wall_0003_right_lower = _wall_cfg(*WALLS[3])
    wall_0004_step_middle = _wall_cfg(*WALLS[4])
    wall_0005_right_upper = _wall_cfg(*WALLS[5])
    wall_0006_top_right = _wall_cfg(*WALLS[6])
    wall_0007_step_join = _wall_cfg(*WALLS[7])

    obstacle_0000_left_island = _obstacle_cfg(*OBSTACLES[0])
    obstacle_0001_center_island = _obstacle_cfg(*OBSTACLES[1])
    obstacle_0002_right_island = _obstacle_cfg(*OBSTACLES[2])

    agent_1: ArticulationCfg = _go2_cfg("{ENV_REGEX_NS}/agent_1", AGENT_1_START_POS)
    agent_2: ArticulationCfg = _go2_cfg("{ENV_REGEX_NS}/agent_2", AGENT_2_START_POS)
    intruder_1: ArticulationCfg = _humanoid_intruder_cfg("{ENV_REGEX_NS}/intruder_1", INTRUDER_START_POS)

    agent_1_front_camera = _front_camera_cfg("{ENV_REGEX_NS}/agent_1/base/front_camera")
    agent_2_front_camera = _front_camera_cfg("{ENV_REGEX_NS}/agent_2/base/front_camera")
    agent_1_lidar = _perception_lidar_cfg("{ENV_REGEX_NS}/agent_1/base")
    agent_2_lidar = _perception_lidar_cfg("{ENV_REGEX_NS}/agent_2/base")
    agent_1_imu = _dog_imu_cfg("{ENV_REGEX_NS}/agent_1/base")
    agent_2_imu = _dog_imu_cfg("{ENV_REGEX_NS}/agent_2/base")

    cam_marker_nw = _cctv_marker("cam_nw")
    cam_marker_ne = _cctv_marker("cam_ne")
    cam_marker_e_upper = _cctv_marker("cam_e_upper")
    cam_marker_e_lower = _cctv_marker("cam_e_lower")
    cam_marker_se = _cctv_marker("cam_se")
    cam_marker_sw = _cctv_marker("cam_sw")

    cam_nw = _cctv_cfg("cam_nw")
    cam_ne = _cctv_cfg("cam_ne")
    cam_e_upper = _cctv_cfg("cam_e_upper")
    cam_e_lower = _cctv_cfg("cam_e_lower")
    cam_se = _cctv_cfg("cam_se")
    cam_sw = _cctv_cfg("cam_sw")

    dome_light = AssetBaseCfg(
        prim_path="/World/Lights/Dome",
        spawn=sim_utils.DomeLightCfg(
            intensity=450.0,
            color=(0.78, 0.84, 1.0),
        ),
    )

    sun = AssetBaseCfg(
        prim_path="/World/Lights/Sun",
        spawn=sim_utils.DistantLightCfg(
            intensity=2600.0,
            color=(1.0, 0.96, 0.9),
            angle=0.45,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=(0.830222, 0.421011, -0.147015, -0.334024),
        ),
    )

    overhead_panel = AssetBaseCfg(
        prim_path="/World/Lights/OverheadPanel",
        spawn=sim_utils.DiskLightCfg(
            intensity=420.0,
            color=(1.0, 0.98, 0.92),
            radius=4.5,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.4, -0.5, 6.0)),
    )
