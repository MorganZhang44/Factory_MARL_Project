"""
astar.py
A* path planning on the 2D occupancy grid from ObstacleMap.
Returns a list of waypoints in world coordinates (metres).
"""
from __future__ import annotations

import heapq
import numpy as np
from typing import Dict, List, Optional, Tuple

from .map_utils import ObstacleMap


# 8-connected grid: (delta_row, delta_col, cost)
_NEIGHBORS = [
    (-1,  0, 1.000), ( 1,  0, 1.000),
    ( 0, -1, 1.000), ( 0,  1, 1.000),
    (-1, -1, 1.414), (-1,  1, 1.414),
    ( 1, -1, 1.414), ( 1,  1, 1.414),
]


def astar(
    obstacle_map: ObstacleMap,
    start_world: Tuple[float, float],
    goal_world:  Tuple[float, float],
) -> List[Tuple[float, float]]:
    """
    Plan a path from start to goal (world coords) using A*.

    Returns:
        List of waypoints (world coords). If no path is found,
        returns [goal_world] as a straight-line fallback.
    """
    grid = obstacle_map.grid
    gs   = obstacle_map.grid_size

    start_rc = obstacle_map.world_to_grid(*start_world)
    goal_rc  = obstacle_map.world_to_grid(*goal_world)

    # Snap start/goal out of obstacles
    start_rc = _snap_free(grid, start_rc)
    goal_rc  = _snap_free(grid, goal_rc)

    if start_rc == goal_rc:
        return [goal_world]

    # --- A* ---
    open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, 0.0, start_rc))

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score:   Dict[Tuple[int, int], float] = {start_rc: 0.0}

    def h(rc: Tuple[int, int]) -> float:
        return float(np.sqrt((rc[0] - goal_rc[0]) ** 2 + (rc[1] - goal_rc[1]) ** 2))

    while open_heap:
        f, g, current = heapq.heappop(open_heap)

        if current == goal_rc:
            return _reconstruct(obstacle_map, came_from, current, goal_world)

        if g > g_score.get(current, float("inf")):
            continue  # stale entry

        for dr, dc, cost in _NEIGHBORS:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < gs and 0 <= nc < gs):
                continue
            if grid[nr, nc] == 1:
                continue
            neighbor = (nr, nc)
            new_g = g + cost
            if new_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = new_g
                came_from[neighbor] = current
                heapq.heappush(open_heap, (new_g + h(neighbor), new_g, neighbor))

    # No path found → straight-line fallback
    return [goal_world]


def _snap_free(
    grid: np.ndarray,
    rc: Tuple[int, int],
) -> Tuple[int, int]:
    """BFS to find nearest free cell."""
    gs = grid.shape[0]
    if grid[rc] == 0:
        return rc
    queue = [rc]
    visited = {rc}
    while queue:
        r, c = queue.pop(0)
        if grid[r, c] == 0:
            return (r, c)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < gs and 0 <= nc < gs and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return rc  # fallback: give up


def _reconstruct(
    obstacle_map: ObstacleMap,
    came_from: Dict[Tuple[int, int], Tuple[int, int]],
    current: Tuple[int, int],
    goal_world: Tuple[float, float],
) -> List[Tuple[float, float]]:
    """Walk back through came_from to build path in world coords."""
    path: List[Tuple[float, float]] = []
    while current in came_from:
        path.append(obstacle_map.grid_to_world(*current))
        current = came_from[current]
    path.reverse()
    path.append(goal_world)
    return path
