"""
map_utils.py
Obstacle definitions extracted from isaac/scenes/warehouse_scene_cfg.py.
Provides collision checking and occupancy grid for A* path planning.

Coordinate frame: world origin at (0, 0), range [-10, +10] m (both axes).
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
# Scene obstacles (from warehouse_scene_cfg.py)
# WALL_THICKNESS=0.3, SCENE_SIZE=20, PILLAR_RADIUS=0.25
# ---------------------------------------------------------------------------

PERIMETER_WALLS: List[RectObstacle] = [
    RectObstacle(cx=0.0,   cy=10.0,  w=20.3, h=0.3),   # North
    RectObstacle(cx=0.0,   cy=-10.0, w=20.3, h=0.3),   # South
    RectObstacle(cx=10.0,  cy=0.0,   w=0.3,  h=20.0),  # East
    RectObstacle(cx=-10.0, cy=0.0,   w=0.3,  h=20.0),  # West
]

INTERIOR_WALLS: List[RectObstacle] = [
    RectObstacle(cx=-3.0, cy=3.0,  w=6.0, h=0.3),  # InteriorWall1 (horizontal)
    RectObstacle(cx=4.0,  cy=-2.5, w=0.3, h=5.0),  # InteriorWall2 (vertical)
    RectObstacle(cx=2.0,  cy=-5.0, w=4.0, h=0.3),  # InteriorWall3 (horizontal)
]

PILLARS: List[CircleObstacle] = [
    CircleObstacle(cx=-5.0, cy=-5.0, r=0.25),  # Pillar1
    CircleObstacle(cx=5.0,  cy=5.0,  r=0.25),  # Pillar2
    CircleObstacle(cx=-2.0, cy=7.0,  r=0.25),  # Pillar3
    CircleObstacle(cx=7.0,  cy=-7.0, r=0.25),  # Pillar4
]

BOXES: List[RectObstacle] = [
    RectObstacle(cx=2.0,  cy=6.0,  w=1.0, h=1.0),  # Box1 + BoxStack1 (same footprint)
    RectObstacle(cx=-6.0, cy=-2.0, w=1.5, h=1.0),  # Box2
    RectObstacle(cx=-3.0, cy=-7.0, w=2.0, h=1.5),  # Box3
    RectObstacle(cx=6.5,  cy=2.0,  w=1.0, h=1.0),  # Box4
]

ALL_RECTS:   List[RectObstacle]   = PERIMETER_WALLS + INTERIOR_WALLS + BOXES
ALL_CIRCLES: List[CircleObstacle] = PILLARS


# ---------------------------------------------------------------------------
# ObstacleMap
# ---------------------------------------------------------------------------

class ObstacleMap:
    """
    Provides:
      - Continuous collision checking (with agent-radius inflation)
      - Pre-built binary occupancy grid for A* planning
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
        self.grid_size    = int(round(2 * map_half / resolution))  # 40 cells @ 0.5 m

        self._rects   = ALL_RECTS
        self._circles = ALL_CIRCLES

        self.grid = self._build_grid()  # (grid_size, grid_size) uint8

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """World (m) → (row, col). Row increases with y."""
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
