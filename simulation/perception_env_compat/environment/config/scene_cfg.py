# Copyright (c) 2026, Multi-Agent Surveillance Project
# Scene configuration: stepped SLAM map with dogs, suspect, CCTV, and onboard sensors.

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

from isaaclab_assets import HUMANOID_CFG, UNITREE_GO2_CFG

from .sensor_cfg import DOG_IMU_CFG
from environment.static_scene_geometry import (
    CAM_HEIGHT,
    get_actor_spawn_points,
    get_camera_positions,
    get_camera_targets,
    get_static_cuboids,
)


SPAWN_POINTS = get_actor_spawn_points()
CAMERA_POSITIONS = get_camera_positions(height=CAM_HEIGHT)
CAMERA_TARGETS = get_camera_targets()
STATIC_CUBOIDS = {cuboid["name"]: cuboid for cuboid in get_static_cuboids(include_floor=True)}

FLOOR_COLOR = (0.72, 0.72, 0.68)
WALL_COLOR = (0.08, 0.08, 0.08)
OBSTACLE_COLOR = (0.36, 0.14, 0.10)
CCTV_PITCH_DEG = 25.0


def _normalize(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    if norm < 1.0e-8:
        return (1.0, 0.0, 0.0)
    return (vec[0] / norm, vec[1] / norm, vec[2] / norm)


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _quat_from_rotmat(
    rot: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
) -> tuple[float, float, float, float]:
    r00, r01, r02 = rot[0]
    r10, r11, r12 = rot[1]
    r20, r21, r22 = rot[2]
    trace = r00 + r11 + r22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r21 - r12) / s
        y = (r02 - r20) / s
        z = (r10 - r01) / s
    elif r00 > r11 and r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        w = (r21 - r12) / s
        x = 0.25 * s
        y = (r01 + r10) / s
        z = (r02 + r20) / s
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        w = (r02 - r20) / s
        x = (r01 + r10) / s
        y = 0.25 * s
        z = (r12 + r21) / s
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        w = (r10 - r01) / s
        x = (r02 + r20) / s
        y = (r12 + r21) / s
        z = 0.25 * s

    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1.0e-8:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / norm, x / norm, y / norm, z / norm)


def _look_at_quat(eye: tuple[float, float, float], target: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Return a world-convention quaternion where +X looks at target and +Z is up-like."""
    forward = _normalize((target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]))
    world_up = (0.0, 0.0, 1.0)
    up_proj = (
        world_up[0] - _dot(world_up, forward) * forward[0],
        world_up[1] - _dot(world_up, forward) * forward[1],
        world_up[2] - _dot(world_up, forward) * forward[2],
    )
    if abs(up_proj[0]) + abs(up_proj[1]) + abs(up_proj[2]) < 1.0e-6:
        up_proj = (0.0, 1.0, 0.0)
    z_axis = _normalize(up_proj)
    y_axis = _normalize(_cross(z_axis, forward))
    z_axis = _normalize(_cross(forward, y_axis))
    rot = (
        (forward[0], y_axis[0], z_axis[0]),
        (forward[1], y_axis[1], z_axis[1]),
        (forward[2], y_axis[2], z_axis[2]),
    )
    return _quat_from_rotmat(rot)


def _look_at_with_fixed_pitch_quat(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    pitch_deg: float,
) -> tuple[float, float, float, float]:
    """Return a quaternion that yaws toward target XY and pitches down by a fixed angle."""
    dx = target[0] - eye[0]
    dy = target[1] - eye[1]
    horizontal_norm = math.hypot(dx, dy)
    if horizontal_norm < 1.0e-6:
        return _look_at_quat(eye, target)

    pitch_rad = math.radians(pitch_deg)
    forward = (
        math.cos(pitch_rad) * dx / horizontal_norm,
        math.cos(pitch_rad) * dy / horizontal_norm,
        -math.sin(pitch_rad),
    )

    world_up = (0.0, 0.0, 1.0)
    y_axis = _normalize(_cross(world_up, forward))
    z_axis = _normalize(_cross(forward, y_axis))
    rot = (
        (forward[0], y_axis[0], z_axis[0]),
        (forward[1], y_axis[1], z_axis[1]),
        (forward[2], y_axis[2], z_axis[2]),
    )
    return _quat_from_rotmat(rot)


def _preview(color: tuple[float, float, float]) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.75)


def _static_scene_cfg(name: str, color: tuple[float, float, float]) -> AssetBaseCfg:
    cuboid = STATIC_CUBOIDS[name]
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=cuboid["size"],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=_preview(color),
            semantic_tags=[("class", "slam_scene")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=cuboid["center"]),
    )


def _dog_cfg(prim_path: str, pos: tuple[float, float, float]) -> ArticulationCfg:
    cfg = UNITREE_GO2_CFG.replace(prim_path=prim_path)
    cfg.init_state = ArticulationCfg.InitialStateCfg(
        pos=pos,
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    )
    cfg.spawn.semantic_tags = [("class", "dog")]
    return cfg


def _suspect_cfg(prim_path: str, pos: tuple[float, float, float]) -> ArticulationCfg:
    cfg = HUMANOID_CFG.replace(prim_path=prim_path)
    cfg.init_state = ArticulationCfg.InitialStateCfg(
        pos=pos,
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    )
    cfg.spawn.semantic_tags = [("class", "suspect")]
    return cfg


def _cctv_cfg(name: str) -> CameraCfg:
    position = CAMERA_POSITIONS[name]
    target = CAMERA_TARGETS[name]
    quat = _look_at_with_fixed_pitch_quat(position, target, pitch_deg=CCTV_PITCH_DEG)
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
            pos=position,
            rot=quat,
            convention="world",
        ),
    )


def _cctv_marker(name: str) -> AssetBaseCfg:
    position = CAMERA_POSITIONS[name]
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}_marker",
        spawn=sim_utils.CuboidCfg(
            size=(0.18, 0.18, 0.18),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.1, 0.1),
                emissive_color=(0.6, 0.0, 0.0),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(position[0], position[1], position[2] + 0.18),
        ),
    )


@configclass
class SurveillanceSceneCfg(InteractiveSceneCfg):
    """Configuration for the stepped SLAM-map surveillance scene."""

    num_envs: int = 1
    env_spacing: float = 12.0
    replicate_physics: bool = False

    floor = _static_scene_cfg("Floor", FLOOR_COLOR)
    wall_0000_back = _static_scene_cfg("Wall_0000_Back", WALL_COLOR)
    wall_0001_left = _static_scene_cfg("Wall_0001_Left", WALL_COLOR)
    wall_0002_front = _static_scene_cfg("Wall_0002_Front", WALL_COLOR)
    wall_0003_right_lower = _static_scene_cfg("Wall_0003_RightLower", WALL_COLOR)
    wall_0004_step_middle = _static_scene_cfg("Wall_0004_StepMiddle", WALL_COLOR)
    wall_0005_right_upper = _static_scene_cfg("Wall_0005_RightUpper", WALL_COLOR)
    wall_0006_top_right = _static_scene_cfg("Wall_0006_TopRight", WALL_COLOR)
    wall_0007_step_join = _static_scene_cfg("Wall_0007_StepJoin", WALL_COLOR)

    obstacle_0000_left_island = _static_scene_cfg("Obstacle_0000_LeftIsland", OBSTACLE_COLOR)
    obstacle_0001_center_island = _static_scene_cfg("Obstacle_0001_CenterIsland", OBSTACLE_COLOR)
    obstacle_0002_right_island = _static_scene_cfg("Obstacle_0002_RightIsland", OBSTACLE_COLOR)

    go2_dog_1: ArticulationCfg = _dog_cfg("{ENV_REGEX_NS}/Go2Dog1", SPAWN_POINTS["dog1"])
    go2_dog_2: ArticulationCfg = _dog_cfg("{ENV_REGEX_NS}/Go2Dog2", SPAWN_POINTS["dog2"])
    suspect: ArticulationCfg = _suspect_cfg("{ENV_REGEX_NS}/Suspect", SPAWN_POINTS["suspect"])

    cam_marker_nw: AssetBaseCfg = _cctv_marker("cam_nw")
    cam_marker_ne: AssetBaseCfg = _cctv_marker("cam_ne")
    cam_marker_e_upper: AssetBaseCfg = _cctv_marker("cam_e_upper")
    cam_marker_e_lower: AssetBaseCfg = _cctv_marker("cam_e_lower")
    cam_marker_se: AssetBaseCfg = _cctv_marker("cam_se")
    cam_marker_sw: AssetBaseCfg = _cctv_marker("cam_sw")

    cam_nw: CameraCfg = _cctv_cfg("cam_nw")
    cam_ne: CameraCfg = _cctv_cfg("cam_ne")
    cam_e_upper: CameraCfg = _cctv_cfg("cam_e_upper")
    cam_e_lower: CameraCfg = _cctv_cfg("cam_e_lower")
    cam_se: CameraCfg = _cctv_cfg("cam_se")
    cam_sw: CameraCfg = _cctv_cfg("cam_sw")

    dog1_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Go2Dog1/base/FrontCam",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.5,
            horizontal_aperture=12.0,
        ),
        width=320,
        height=240,
        data_types=["rgb", "depth", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        update_period=0.1,
        offset=CameraCfg.OffsetCfg(
            pos=(0.3, 0.0, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )

    dog2_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Go2Dog2/base/FrontCam",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.5,
            horizontal_aperture=12.0,
        ),
        width=320,
        height=240,
        data_types=["rgb", "depth", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        update_period=0.1,
        offset=CameraCfg.OffsetCfg(
            pos=(0.3, 0.0, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )

    dog1_lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Go2Dog1/base",
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
        mesh_prim_paths=["/World/LocalizationStaticMesh"],
    )

    dog2_lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Go2Dog2/base",
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
        mesh_prim_paths=["/World/LocalizationStaticMesh"],
    )

    go2_dog_1_imu = DOG_IMU_CFG.replace(prim_path="{ENV_REGEX_NS}/Go2Dog1/base")
    go2_dog_2_imu = DOG_IMU_CFG.replace(prim_path="{ENV_REGEX_NS}/Go2Dog2/base")

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
