# Copyright (c) 2026, Multi-Agent Surveillance Project
# Scene configuration converted from simulation/assets/scenes/slam_scene_with_actors.usda.

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import H1_MINIMAL_CFG, UNITREE_GO2_CFG


FLOOR_COLOR = (0.72, 0.72, 0.68)
WALL_COLOR = (0.08, 0.08, 0.08)
OBSTACLE_COLOR = (0.36, 0.14, 0.10)

AGENT_1_START_POS = (-4.0, -3.0, 0.4000000059604645)
AGENT_2_START_POS = (-4.0, 2.2, 0.4000000059604645)
INTRUDER_START_POS = (4.8, 1.0, 1.0499999523162842)

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


def _h1_intruder_cfg(prim_path: str, pos: tuple[float, float, float]) -> ArticulationCfg:
    cfg = H1_MINIMAL_CFG.replace(prim_path=prim_path)
    cfg.init_state = ArticulationCfg.InitialStateCfg(
        pos=pos,
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    )
    cfg.spawn.semantic_tags = [("class", "intruder")]
    return cfg


def _front_camera_cfg(prim_path: str) -> CameraCfg:
    return CameraCfg(
        prim_path=prim_path,
        spawn=sim_utils.PinholeCameraCfg(
            clipping_range=(0.05, 1000.0),
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
        ),
        width=320,
        height=240,
        data_types=["rgb"],
        update_period=0.067,
        offset=CameraCfg.OffsetCfg(
            pos=(0.34, 0.0, 0.12),
            rot=(0.5, 0.5, -0.5, -0.5),
            convention="world",
        ),
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
    actors and front cameras come from ``slam_scene_with_actors.usda``.
    Lidar prims are intentionally omitted for now.
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
    intruder_1: ArticulationCfg = _h1_intruder_cfg("{ENV_REGEX_NS}/intruder_1", INTRUDER_START_POS)

    agent_1_front_camera = _front_camera_cfg("{ENV_REGEX_NS}/agent_1/base/front_camera")
    agent_2_front_camera = _front_camera_cfg("{ENV_REGEX_NS}/agent_2/base/front_camera")

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
