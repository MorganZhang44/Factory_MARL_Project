# Copyright (c) 2026, Multi-Agent Surveillance Project
# Unified scene configuration for the MARL pursuit system.
#
# This file is the SINGLE SOURCE OF TRUTH for the scene.
# It contains two sections:
#
#   SECTION A  ━  2-D map data (pure Python, no Isaac Sim needed)
#                 Imported by the MARL training & A* planner.
#
#   SECTION B  ━  Isaac Lab scene config (requires Isaac Sim)
#                 Loaded by the Isaac Sim simulation.
#
# Usage from MARL code:
#   from marl.utils.map_utils import ObstacleMap      ← (map_utils re-exports this)
#
# Usage from Isaac Sim:
#   from isaac.scenes.slam_scene_cfg import SlamSceneCfg

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ════════════════════════════════════════════════════════════════════════════
# SECTION A — 2-D map data  (pure Python, always importable)
# ════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Scene geometry constants
# ---------------------------------------------------------------------------

MAP_HALF        = 10.0   # m  — map spans [-10, +10] on both axes
WALL_THICKNESS  = 0.3    # m

# Initial positions aligned with the Isaac Sim scene (x, y only)
AGENT_1_POS_2D  = (-2.0, -2.0)
AGENT_2_POS_2D  = (-2.0,  1.6)
INTRUDER_POS_2D = ( 2.0, -0.5)


# ---------------------------------------------------------------------------
# Obstacle primitives
# ---------------------------------------------------------------------------

@dataclass
class RectObstacle:
    """Axis-aligned rectangle.  cx, cy = centre;  w, h = full width/height."""
    cx: float
    cy: float
    w: float
    h: float

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (abs(x - self.cx) <= self.w / 2 + margin and
                abs(y - self.cy) <= self.h / 2 + margin)


@dataclass
class CircleObstacle:
    """Circular obstacle (pillars).  cx, cy = centre;  r = radius."""
    cx: float
    cy: float
    r: float

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= (self.r + margin) ** 2


# ---------------------------------------------------------------------------
# Scene obstacles
# ---------------------------------------------------------------------------

# Perimeter boundary — always present
PERIMETER_WALLS: List[RectObstacle] = [
    RectObstacle(cx=0.0,   cy=10.0,  w=20.3, h=WALL_THICKNESS),  # North
    RectObstacle(cx=0.0,   cy=-10.0, w=20.3, h=WALL_THICKNESS),  # South
    RectObstacle(cx=10.0,  cy=0.0,   w=WALL_THICKNESS, h=20.0),  # East
    RectObstacle(cx=-10.0, cy=0.0,   w=WALL_THICKNESS, h=20.0),  # West
]

# Interior obstacles — derived from the SLAM-scanned USD scene.
# TODO: populate once slam_scene.usda obstacle data has been exported
#       (bounding-box list or 2-D occupancy grid from the SLAM pipeline).
# Example:
#   RectObstacle(cx=2.5, cy=1.0, w=1.2, h=0.3)   ← interior wall segment
#   CircleObstacle(cx=-3.0, cy=4.0, r=0.25)       ← pillar
INTERIOR_WALLS: List[RectObstacle]   = []
PILLARS:        List[CircleObstacle] = []
BOXES:          List[RectObstacle]   = []

ALL_RECTS:   List[RectObstacle]   = PERIMETER_WALLS + INTERIOR_WALLS + BOXES
ALL_CIRCLES: List[CircleObstacle] = PILLARS


# ---------------------------------------------------------------------------
# ObstacleMap — occupancy grid + continuous collision checking for A*
# ---------------------------------------------------------------------------

class ObstacleMap:
    """
    Provides:
      - Continuous collision checking (with agent-radius inflation)
      - Pre-built binary occupancy grid for A* path planning
    """

    def __init__(
        self,
        map_half:     float = MAP_HALF,
        resolution:   float = 0.5,
        agent_radius: float = 0.3,
    ):
        self.map_half     = map_half
        self.resolution   = resolution
        self.agent_radius = agent_radius
        self.grid_size    = int(round(2 * map_half / resolution))

        self._rects   = ALL_RECTS
        self._circles = ALL_CIRCLES
        self.grid     = self._build_grid()   # (grid_size, grid_size) uint8

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """World (m) → (row, col).  Row increases with +y."""
        col = int((x + self.map_half) / self.resolution)
        row = int((y + self.map_half) / self.resolution)
        col = int(np.clip(col, 0, self.grid_size - 1))
        row = int(np.clip(row, 0, self.grid_size - 1))
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Grid cell centre → world (m)."""
        x = (col + 0.5) * self.resolution - self.map_half
        y = (row + 0.5) * self.resolution - self.map_half
        return x, y

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def _build_grid(self) -> np.ndarray:
        """Binary grid: 1 = obstacle (inflated by agent_radius)."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        m = self.agent_radius
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                wx, wy = self.grid_to_world(row, col)
                for obs in self._rects:
                    if obs.contains_point(wx, wy, m):
                        grid[row, col] = 1
                        break
                else:
                    for obs in self._circles:
                        if obs.contains_point(wx, wy, m):
                            grid[row, col] = 1
                            break
        return grid

    # ------------------------------------------------------------------
    # Collision checking (continuous, used at every env step)
    # ------------------------------------------------------------------

    def is_collision(self, x: float, y: float) -> bool:
        if abs(x) >= self.map_half or abs(y) >= self.map_half:
            return True
        m = self.agent_radius
        for obs in self._rects:
            if obs.contains_point(x, y, m):
                return True
        for obs in self._circles:
            if obs.contains_point(x, y, m):
                return True
        return False

    def get_grid(self) -> np.ndarray:
        return self.grid.copy()


# ════════════════════════════════════════════════════════════════════════════
# SECTION B — Isaac Lab scene config  (requires Isaac Sim / Isaac Lab)
# ════════════════════════════════════════════════════════════════════════════

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns
    from isaaclab.utils import configclass
    from isaaclab_assets.robots.unitree import H1_MINIMAL_CFG, UNITREE_GO2_CFG

    _ISAAC_AVAILABLE = True

except ImportError:
    # Isaac Lab not installed — Section A (pure 2-D map data) still works.
    _ISAAC_AVAILABLE = False

if _ISAAC_AVAILABLE:

    PROJECT_ROOT   = Path(__file__).resolve().parents[2]
    SLAM_SCENE_USD = PROJECT_ROOT / "isaac" / "assets" / "scenes" / "slam_scene.usda"

    RPLIDAR_S2E_USD = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/5.1/Isaac/Sensors/Slamtec/RPLIDAR_S2E/Slamtec_RPLIDAR_S2E.usd"
    )

    # 3-D spawn positions (x, y, z) — derived from 2-D constants above
    AGENT_1_POS  = (*AGENT_1_POS_2D,  0.42)
    AGENT_2_POS  = (*AGENT_2_POS_2D,  0.42)
    INTRUDER_POS = (*INTRUDER_POS_2D, 1.05)

    DOG_JOINT_POS = {
        ".*L_hip_joint":      0.1,
        ".*R_hip_joint":     -0.1,
        "F[L,R]_thigh_joint": 0.8,
        "R[L,R]_thigh_joint": 1.0,
        ".*_calf_joint":     -1.5,
    }

    SEMANTIC_LABELS = {
        "class:intruder":   (255,  50,  50, 255),
        "class:dog":        ( 50, 150, 255, 255),
        "class:slam_scene": (150, 150, 150, 255),
    }

    # ------------------------------------------------------------------
    # Helper factories
    # ------------------------------------------------------------------

    def _go2_cfg(prim_path: str, pos: tuple) -> ArticulationCfg:
        cfg = UNITREE_GO2_CFG.replace(prim_path=prim_path)
        cfg.init_state = ArticulationCfg.InitialStateCfg(
            pos=pos,
            joint_pos=DOG_JOINT_POS,
            joint_vel={".*": 0.0},
        )
        cfg.spawn.semantic_tags = [("class", "dog")]
        return cfg

    def _h1_intruder_cfg(prim_path: str, pos: tuple) -> ArticulationCfg:
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

    # ------------------------------------------------------------------
    # Scene class
    # ------------------------------------------------------------------

    @configclass
    class SlamSceneCfg(InteractiveSceneCfg):
        """Configuration for the imported SLAM scene.

        Layout:
        - Imported SLAM-scanned scene from slam_scene.usda
        - Two Unitree Go2 robot dogs with front camera + LiDAR
        - One Unitree H1 intruder (humanoid)
        - Visible RPLIDAR_S2E model for viewport inspection
        - Dome + distant lighting
        """

        num_envs:          int   = 1
        env_spacing:       float = 10.0
        replicate_physics: bool  = False

        # Environment — imported SLAM scene
        slam_scene = AssetBaseCfg(
            prim_path="/World/SlamScene",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(SLAM_SCENE_USD),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                semantic_tags=[("class", "slam_scene")],
            ),
        )

        # Actors
        agent_1:    ArticulationCfg = _go2_cfg("{ENV_REGEX_NS}/agent_1",   AGENT_1_POS)
        agent_2:    ArticulationCfg = _go2_cfg("{ENV_REGEX_NS}/agent_2",   AGENT_2_POS)
        intruder_1: ArticulationCfg = _h1_intruder_cfg("{ENV_REGEX_NS}/intruder_1", INTRUDER_POS)

        # Onboard cameras
        agent_1_front_camera = _front_camera_cfg("{ENV_REGEX_NS}/agent_1/base/front_camera")
        agent_2_front_camera = _front_camera_cfg("{ENV_REGEX_NS}/agent_2/base/front_camera")

        # LiDAR ray casters
        agent_1_lidar = _raycast_lidar_cfg("{ENV_REGEX_NS}/agent_1/base")
        agent_2_lidar = _raycast_lidar_cfg("{ENV_REGEX_NS}/agent_2/base")

        # Visible RPLIDAR models (viewport only)
        agent_1_lidar_model = _rplidar_visual_cfg("{ENV_REGEX_NS}/agent_1/base/front_lidar_model")
        agent_2_lidar_model = _rplidar_visual_cfg("{ENV_REGEX_NS}/agent_2/base/front_lidar_model")

        # Lighting
        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=800.0, color=(0.9, 0.9, 1.0)),
        )

        distant_light = AssetBaseCfg(
            prim_path="/World/DistantLight",
            spawn=sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 0.95, 0.85)),
            init_state=AssetBaseCfg.InitialStateCfg(rot=(0.866, 0.0, 0.5, 0.0)),
        )
