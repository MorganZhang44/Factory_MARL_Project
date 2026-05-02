"""
map_utils.py
Utilities for 2D map representation and obstacle handling.
Obstacle definitions extracted from simulation/scenes/slam_scene_cfg.py.
Provides collision checking and occupancy grid for A* path planning.

Coordinate frame: world origin at (0, 0), range [-10, +10] m (both axes).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Map Bounds & Parameters
# ---------------------------------------------------------------------------
MAP_HALF = 10.0         # m (map bounds: x & y in [-10.0, 10.0])
WALL_THICKNESS = 0.3    # m (default thickness)

# Initial 2D Starting Positions (extracted from new_scene.py)
AGENT_1_POS_2D = (-4.0, -3.0)
AGENT_2_POS_2D = (-4.0, 2.2)
INTRUDER_POS_2D = (4.8, 1.0)


# ---------------------------------------------------------------------------
# Obstacle Primitive Data Classes
# ---------------------------------------------------------------------------
@dataclass
class RectObstacle:
    """Axis-aligned rectangular obstacle. cx, cy = center; w, h = total width/height."""
    cx: float
    cy: float
    w: float
    h: float

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (abs(x - self.cx) <= self.w / 2 + margin and
                abs(y - self.cy) <= self.h / 2 + margin)

@dataclass
class CircleObstacle:
    """Circular obstacle. cx, cy = center; r = radius."""
    cx: float
    cy: float
    r: float

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= (self.r + margin) ** 2


# ---------------------------------------------------------------------------
# Obstacles Definitions (From new_scene.py)
# ---------------------------------------------------------------------------

# Extracted from new_scene.py WALLS + Solid padding for outside space
PERIMETER_WALLS: List[RectObstacle] = [
    # Actual walls of the room
    RectObstacle(cx=-0.85, cy=3.45,  w=9.52, h=0.16),  # Wall_0000_Back
    RectObstacle(cx=-5.50, cy=-0.65, w=0.16, h=8.16),  # Wall_0001_Left
    RectObstacle(cx=-0.95, cy=-4.65, w=9.18, h=0.16),  # Wall_0002_Front
    RectObstacle(cx=3.75,  cy=-2.85, w=0.16, h=4.08),  # Wall_0003_RightLower
    RectObstacle(cx=5.00,  cy=-0.95, w=3.06, h=0.16),  # Wall_0004_StepMiddle
    RectObstacle(cx=6.25,  cy=1.35,  w=0.16, h=4.93),  # Wall_0005_RightUpper
    RectObstacle(cx=5.15,  cy=3.45,  w=2.89, h=0.16),  # Wall_0006_TopRight
    RectObstacle(cx=3.75,  cy=2.65,  w=0.16, h=2.125), # Wall_0007_StepJoin
    
    # 🚨 SOLID OUTER PADDING: prevents clipping or walking outside the room bounds
    RectObstacle(cx=0.0,  cy=7.0,   w=22.0, h=7.0),    # Top Solid Bound
    RectObstacle(cx=0.0,  cy=-7.5,  w=22.0, h=5.5),    # Bottom Solid Bound
    RectObstacle(cx=-8.0, cy=0.0,   w=5.0,  h=22.0),   # Left Solid Bound
    RectObstacle(cx=8.5,  cy=0.0,   w=4.0,  h=22.0),   # Right Solid Bound
    
    # Bottom-right 'cutout' area filling
    RectObstacle(cx=6.0,  cy=-3.0,  w=4.5,  h=4.5),    # Bottom-Right corner padding
]

# Extracted from new_scene.py OBSTACLES
INTERIOR_WALLS: List[RectObstacle] = [
    RectObstacle(cx=-2.30, cy=-0.85, w=0.425, h=1.275), # LeftIsland
    RectObstacle(cx=0.40,  cy=-0.95, w=0.425, h=1.375), # CenterIsland
    RectObstacle(cx=2.95,  cy=-0.65, w=0.45,  h=1.40),  # RightIsland
]

BOXES: List[RectObstacle] = []
PILLARS: List[CircleObstacle] = []

ALL_RECTS: List[RectObstacle] = PERIMETER_WALLS + INTERIOR_WALLS + BOXES
ALL_CIRCLES: List[CircleObstacle] = PILLARS


# ---------------------------------------------------------------------------
# Occupancy Grid / Collision Map
# ---------------------------------------------------------------------------
class ObstacleMap:
    def __init__(
        self,
        map_half: float = MAP_HALF,
        resolution: float = 0.5,
        agent_radius: float = 0.3,
    ):
        self.map_half = map_half
        self.resolution = resolution
        self.agent_radius = agent_radius
        self.grid_size = int(round(2 * map_half / resolution))

        self._rects = ALL_RECTS
        self._circles = ALL_CIRCLES
        
        self.grid = self._build_grid()

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        col = int((x + self.map_half) / self.resolution)
        row = int((y + self.map_half) / self.resolution)
        col = int(np.clip(col, 0, self.grid_size - 1))
        row = int(np.clip(row, 0, self.grid_size - 1))
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        x = (col + 0.5) * self.resolution - self.map_half
        y = (row + 0.5) * self.resolution - self.map_half
        return x, y

    def _build_grid(self) -> np.ndarray:
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

    def ray_cast(
        self,
        ox: float, oy: float,
        angle: float,
        max_range: float = 10.0,
        step: float = 0.2,
    ) -> float:
        """
        Cast a single ray from (ox, oy) in the given angle (radians).
        Returns the normalized distance [0, 1] to the nearest obstacle,
        where 1.0 = max_range (clear) and 0.0 = obstacle right next to agent.
        """
        dx = np.cos(angle) * step
        dy = np.sin(angle) * step
        x, y = ox, oy
        dist = 0.0
        while dist < max_range:
            x += dx
            y += dy
            dist += step
            if self.is_collision(x, y):
                return dist / max_range  # normalised
        return 1.0  # no obstacle found → full range
