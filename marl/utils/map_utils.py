"""
map_utils.py
Obstacle definitions for the MARL pursuit environment.

Scene: SLAM-scanned scene (isaac/scenes/slam_scene_cfg.py)
       The scene geometry is contained in slam_scene.usda (imported USD).

Coordinate frame:
    World origin at (0, 0); x ∈ [-10, +10] m, y ∈ [-10, +10] m.

NOTE ON INTERIOR OBSTACLES:
    The new SLAM scene loads obstacles from a pre-built USD file
    (slam_scene.usda). Since the exact obstacle coordinates are
    embedded in the USD rather than defined as Python constants,
    the interior obstacle list below is intentionally left EMPTY.

    Once the SLAM map is exported (e.g. as a 2-D occupancy grid or
    a list of bounding boxes), populate INTERIOR_OBSTACLES and
    PILLARS accordingly to give the A* planner accurate collision
    information.

    For now, A* operates with only perimeter walls, which is
    conservative but correct (it will never plan through real walls).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Obstacle primitives
# ---------------------------------------------------------------------------

@dataclass
class RectObstacle:
    """Axis-aligned rectangle. cx, cy = centre; w, h = full width/height."""
    cx: float
    cy: float
    w: float
    h: float

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (abs(x - self.cx) <= self.w / 2 + margin and
                abs(y - self.cy) <= self.h / 2 + margin)


@dataclass
class CircleObstacle:
    """Circular obstacle (pillars). cx, cy = centre; r = radius."""
    cx: float
    cy: float
    r: float

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= (self.r + margin) ** 2


# ---------------------------------------------------------------------------
# Scene obstacles (SLAM scene)
# ---------------------------------------------------------------------------

# Perimeter boundary — 20 × 20 m enclosure (always present)
WALL_THICKNESS = 0.3

PERIMETER_WALLS: List[RectObstacle] = [
    RectObstacle(cx=0.0,   cy=10.0,  w=20.3, h=WALL_THICKNESS),   # North
    RectObstacle(cx=0.0,   cy=-10.0, w=20.3, h=WALL_THICKNESS),   # South
    RectObstacle(cx=10.0,  cy=0.0,   w=WALL_THICKNESS, h=20.0),   # East
    RectObstacle(cx=-10.0, cy=0.0,   w=WALL_THICKNESS, h=20.0),   # West
]

# ── Interior obstacles (from SLAM USD) ────────────────────────────────────
# TODO: populate once slam_scene.usda obstacle data is exported.
# Example format:
#   RectObstacle(cx=<x>, cy=<y>, w=<width>, h=<height>)
#   CircleObstacle(cx=<x>, cy=<y>, r=<radius>)

INTERIOR_WALLS: List[RectObstacle] = []   # ← fill from SLAM map data
PILLARS:        List[CircleObstacle] = [] # ← fill from SLAM map data
BOXES:          List[RectObstacle] = []   # ← fill from SLAM map data

# Combined lists used by ObstacleMap
ALL_RECTS:   List[RectObstacle]   = PERIMETER_WALLS + INTERIOR_WALLS + BOXES
ALL_CIRCLES: List[CircleObstacle] = PILLARS


# ---------------------------------------------------------------------------
# ObstacleMap
# ---------------------------------------------------------------------------

class ObstacleMap:
    """
    Provides:
      - Continuous collision checking (with agent-radius inflation)
      - Pre-built binary occupancy grid for A* path planning
    """

    def __init__(
        self,
        map_half: float = 10.0,
        resolution: float = 0.5,
        agent_radius: float = 0.3,
    ):
        self.map_half     = map_half
        self.resolution   = resolution
        self.agent_radius = agent_radius
        self.grid_size    = int(round(2 * map_half / resolution))

        self._rects   = ALL_RECTS
        self._circles = ALL_CIRCLES

        self.grid = self._build_grid()  # (grid_size, grid_size) uint8

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """World (m) → (row, col). Row increases with +y."""
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
