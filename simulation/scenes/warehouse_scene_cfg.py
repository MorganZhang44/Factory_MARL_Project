# Copyright (c) 2026, Multi-Agent Surveillance Project
# Scene configuration: warehouse/courtyard scene with robots, suspect, and sensors.

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# Import robot configs from isaaclab_assets
# NOTE: isaaclab_assets becomes a top-level importable package after AppLauncher boots
from isaaclab_assets import UNITREE_GO2_CFG, HUMANOID_CFG

from .sensor_cfg import SURVEILLANCE_CAM_CFG, DOG_CAM_CFG, DOG_LIDAR_CFG


# ===========================================================================
# Scene layout constants
# ===========================================================================
SCENE_SIZE = 20.0           # 20m × 20m area
WALL_HEIGHT = 3.0           # 3m high walls
WALL_THICKNESS = 0.3        # 30cm thick walls
CAM_HEIGHT = 4.5            # surveillance cameras at 4.5m height
PILLAR_RADIUS = 0.25        # 25cm radius pillars
PILLAR_HEIGHT = 3.0
BOX_SIZES = [
    (1.0, 1.0, 1.0),       # small box
    (1.5, 1.0, 0.8),       # medium box
    (2.0, 1.5, 1.2),       # large box
]

# Semantic class labels for detection
SEMANTIC_LABELS = {
    "class:suspect": (255, 50, 50, 255),    # Red
    "class:dog": (50, 150, 255, 255),       # Blue
    "class:wall": (150, 150, 150, 255),     # Gray
    "class:ground": (100, 100, 100, 255),   # Dark gray
    "class:pillar": (180, 160, 120, 255),   # Tan
    "class:box": (139, 90, 43, 255),        # Brown
}


@configclass
class SurveillanceSceneCfg(InteractiveSceneCfg):
    """Configuration for the multi-agent surveillance scene.

    Layout: A 20m×20m warehouse/courtyard with:
    - Perimeter walls
    - Interior pillars and box obstacles
    - 4 overhead surveillance cameras at corners
    - 2 Unitree Go2 robot dogs with cameras + LiDAR
    - 1 humanoid suspect
    """

    num_envs: int = 1
    env_spacing: float = 25.0
    replicate_physics: bool = False   # single env, no need for replication

    # ------------------------------------------------------------------
    # Terrain: flat ground
    # ------------------------------------------------------------------
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # ------------------------------------------------------------------
    # Perimeter walls (4 walls forming the enclosure)
    # ------------------------------------------------------------------
    wall_north = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallNorth",
        spawn=sim_utils.CuboidCfg(
            size=(SCENE_SIZE + WALL_THICKNESS, WALL_THICKNESS, WALL_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.55, 0.55),
            ),
            semantic_tags=[("class", "wall")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, SCENE_SIZE / 2, WALL_HEIGHT / 2),
        ),
    )

    wall_south = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallSouth",
        spawn=sim_utils.CuboidCfg(
            size=(SCENE_SIZE + WALL_THICKNESS, WALL_THICKNESS, WALL_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.55, 0.55),
            ),
            semantic_tags=[("class", "wall")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, -SCENE_SIZE / 2, WALL_HEIGHT / 2),
        ),
    )

    wall_east = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallEast",
        spawn=sim_utils.CuboidCfg(
            size=(WALL_THICKNESS, SCENE_SIZE, WALL_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.55, 0.55),
            ),
            semantic_tags=[("class", "wall")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(SCENE_SIZE / 2, 0.0, WALL_HEIGHT / 2),
        ),
    )

    wall_west = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallWest",
        spawn=sim_utils.CuboidCfg(
            size=(WALL_THICKNESS, SCENE_SIZE, WALL_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.55, 0.55),
            ),
            semantic_tags=[("class", "wall")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-SCENE_SIZE / 2, 0.0, WALL_HEIGHT / 2),
        ),
    )

    # ------------------------------------------------------------------
    # Interior walls (create corridors / partial occlusion)
    # ------------------------------------------------------------------
    interior_wall_1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/InteriorWall1",
        spawn=sim_utils.CuboidCfg(
            size=(6.0, WALL_THICKNESS, WALL_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.58, 0.55),
            ),
            semantic_tags=[("class", "wall")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-3.0, 3.0, WALL_HEIGHT / 2),
        ),
    )

    interior_wall_2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/InteriorWall2",
        spawn=sim_utils.CuboidCfg(
            size=(WALL_THICKNESS, 5.0, WALL_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.58, 0.55),
            ),
            semantic_tags=[("class", "wall")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(4.0, -2.5, WALL_HEIGHT / 2),
        ),
    )

    interior_wall_3 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/InteriorWall3",
        spawn=sim_utils.CuboidCfg(
            size=(4.0, WALL_THICKNESS, WALL_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.58, 0.55),
            ),
            semantic_tags=[("class", "wall")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(2.0, -5.0, WALL_HEIGHT / 2),
        ),
    )

    # ------------------------------------------------------------------
    # Pillars (structural obstacles)
    # ------------------------------------------------------------------
    pillar_1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pillar1",
        spawn=sim_utils.CylinderCfg(
            radius=PILLAR_RADIUS,
            height=PILLAR_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.7, 0.65, 0.55),
            ),
            semantic_tags=[("class", "pillar")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-5.0, -5.0, PILLAR_HEIGHT / 2),
        ),
    )

    pillar_2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pillar2",
        spawn=sim_utils.CylinderCfg(
            radius=PILLAR_RADIUS,
            height=PILLAR_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.7, 0.65, 0.55),
            ),
            semantic_tags=[("class", "pillar")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(5.0, 5.0, PILLAR_HEIGHT / 2),
        ),
    )

    pillar_3 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pillar3",
        spawn=sim_utils.CylinderCfg(
            radius=PILLAR_RADIUS,
            height=PILLAR_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.7, 0.65, 0.55),
            ),
            semantic_tags=[("class", "pillar")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-2.0, 7.0, PILLAR_HEIGHT / 2),
        ),
    )

    pillar_4 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pillar4",
        spawn=sim_utils.CylinderCfg(
            radius=PILLAR_RADIUS,
            height=PILLAR_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.7, 0.65, 0.55),
            ),
            semantic_tags=[("class", "pillar")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(7.0, -7.0, PILLAR_HEIGHT / 2),
        ),
    )

    # ------------------------------------------------------------------
    # Boxes / crates (obstacles providing cover)
    # ------------------------------------------------------------------
    box_1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Box1",
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZES[0],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.35, 0.15),
            ),
            semantic_tags=[("class", "box")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(2.0, 6.0, 0.5),
        ),
    )

    box_2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Box2",
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZES[1],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.50, 0.32, 0.12),
            ),
            semantic_tags=[("class", "box")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-6.0, -2.0, 0.4),
        ),
    )

    box_3 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Box3",
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZES[2],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.45, 0.30, 0.10),
            ),
            semantic_tags=[("class", "box")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-3.0, -7.0, 0.6),
        ),
    )

    box_4 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Box4",
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZES[0],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.35, 0.15),
            ),
            semantic_tags=[("class", "box")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(6.5, 2.0, 0.5),
        ),
    )

    box_stack_1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BoxStack1",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.50, 0.38, 0.18),
            ),
            semantic_tags=[("class", "box")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(2.0, 6.0, 1.4),   # stacked on top of box_1
        ),
    )

    # ------------------------------------------------------------------
    # Robot: Unitree Go2 Dog 1
    # ------------------------------------------------------------------
    go2_dog_1: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Go2Dog1",
    )
    go2_dog_1.init_state = ArticulationCfg.InitialStateCfg(
        pos=(3.0, 3.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    )
    go2_dog_1.spawn.semantic_tags = [("class", "dog")]
    # ------------------------------------------------------------------
    # Robot: Unitree Go2 Dog 2
    # ------------------------------------------------------------------
    go2_dog_2: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Go2Dog2",
    )
    go2_dog_2.init_state = ArticulationCfg.InitialStateCfg(
        pos=(-3.0, -3.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    )
    go2_dog_2.spawn.semantic_tags = [("class", "dog")]
    # ------------------------------------------------------------------
    # Suspect: Humanoid figure
    # ------------------------------------------------------------------
    suspect: ArticulationCfg = HUMANOID_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Suspect",
    )
    suspect.init_state = ArticulationCfg.InitialStateCfg(
        pos=(5.0, 0.0, 1.34),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    )
    suspect.spawn.semantic_tags = [("class", "suspect")]
    # ------------------------------------------------------------------
    # Visible Camera Markers (so cameras are visible in viewport)
    # ------------------------------------------------------------------
    cam_marker_ne: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CamMarkerNE",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.3),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.1, 0.1),  # red
                emissive_color=(0.5, 0.0, 0.0),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(9.0, 9.0, CAM_HEIGHT + 0.3)),
    )
    cam_marker_nw: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CamMarkerNW",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.3),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.1, 0.1),
                emissive_color=(0.5, 0.0, 0.0),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-9.0, 9.0, CAM_HEIGHT + 0.3)),
    )
    cam_marker_se: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CamMarkerSE",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.3),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.1, 0.1),
                emissive_color=(0.5, 0.0, 0.0),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(9.0, -9.0, CAM_HEIGHT + 0.3)),
    )
    cam_marker_sw: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CamMarkerSW",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.3),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.1, 0.1),
                emissive_color=(0.5, 0.0, 0.0),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-9.0, -9.0, CAM_HEIGHT + 0.3)),
    )

    # ------------------------------------------------------------------
    # Surveillance Cameras (4 corners, looking inward & downward)
    # ------------------------------------------------------------------
    # Camera convention "world": forward = +X, up = +Z
    # We use quaternions to orient cameras to look inward and down

    # NE corner → looking SW and down
    cam_ne = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CamNE",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            horizontal_aperture=20.955,
        ),
        width=640,
        height=480,
        data_types=["rgb", "depth", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        update_period=0.1,
        offset=CameraCfg.OffsetCfg(
            pos=(9.0, 9.0, CAM_HEIGHT),
            rot=(-0.377, -0.156, -0.065, 0.911),   # pointing at (0,0,0)
            convention="world",
        ),
    )

    # NW corner → looking SE and down
    cam_nw = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CamNW",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            horizontal_aperture=20.955,
        ),
        width=640,
        height=480,
        data_types=["rgb", "depth", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        update_period=0.1,
        offset=CameraCfg.OffsetCfg(
            pos=(-9.0, 9.0, CAM_HEIGHT),
            rot=(0.911, 0.065, 0.156, -0.377),   # pointing at (0,0,0)
            convention="world",
        ),
    )

    # SE corner → looking NW and down
    cam_se = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CamSE",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            horizontal_aperture=20.955,
        ),
        width=640,
        height=480,
        data_types=["rgb", "depth", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        update_period=0.1,
        offset=CameraCfg.OffsetCfg(
            pos=(9.0, -9.0, CAM_HEIGHT),
            rot=(0.377, -0.156, 0.065, 0.911),   # pointing at (0,0,0)
            convention="world",
        ),
    )

    # SW corner → looking NE and down
    cam_sw = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CamSW",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            horizontal_aperture=20.955,
        ),
        width=640,
        height=480,
        data_types=["rgb", "depth", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        update_period=0.1,
        offset=CameraCfg.OffsetCfg(
            pos=(-9.0, -9.0, CAM_HEIGHT),
            rot=(0.911, -0.065, 0.156, 0.377),   # pointing at (0,0,0)
            convention="world",
        ),
    )

    # ------------------------------------------------------------------
    # Dog 1: onboard camera (front-facing, 120° FOV)
    # ------------------------------------------------------------------
    dog1_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Go2Dog1/base/FrontCam",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.5,
            horizontal_aperture=12.0,
        ),
        width=640,
        height=480,
        data_types=["rgb", "depth", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        update_period=0.067,
        offset=CameraCfg.OffsetCfg(
            pos=(0.3, 0.0, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )

    # ------------------------------------------------------------------
    # Dog 2: onboard camera (front-facing, 120° FOV)
    # ------------------------------------------------------------------
    dog2_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Go2Dog2/base/FrontCam",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.5,
            horizontal_aperture=12.0,
        ),
        width=640,
        height=480,
        data_types=["rgb", "depth", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        update_period=0.067,
        offset=CameraCfg.OffsetCfg(
            pos=(0.3, 0.0, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )

    # ------------------------------------------------------------------
    # Dog 1: LiDAR (360° × 90° FOV)
    # ------------------------------------------------------------------
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
        mesh_prim_paths=["/World/ground"],
    )

    # ------------------------------------------------------------------
    # Dog 2: LiDAR (360° × 90° FOV)
    # ------------------------------------------------------------------
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
        mesh_prim_paths=["/World/ground"],
    )

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
            rot=(0.866, 0.0, 0.5, 0.0),  # angled sunlight
        ),
    )
