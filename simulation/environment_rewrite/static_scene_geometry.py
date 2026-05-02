"""Shared static scene geometry derived from the provided SLAM scene config."""

from __future__ import annotations

import heapq
import math


FLOOR_CENTER = (-0.85, -0.65, -0.02)
FLOOR_SIZE = (16.0, 9.5, 0.02)

_STATIC_CUBOIDS = (
    {"name": "Floor", "label": "floor", "center": FLOOR_CENTER, "size": FLOOR_SIZE},
    {"name": "Wall_0000_Back", "label": "wall", "center": (-0.85, 3.45, 0.9), "size": (9.52, 0.16, 1.8)},
    {"name": "Wall_0001_Left", "label": "wall", "center": (-5.5, -0.65, 0.9), "size": (0.16, 8.16, 1.8)},
    {"name": "Wall_0002_Front", "label": "wall", "center": (-0.95, -4.65, 0.9), "size": (9.18, 0.16, 1.8)},
    {"name": "Wall_0003_RightLower", "label": "wall", "center": (3.75, -2.85, 0.9), "size": (0.16, 4.08, 1.8)},
    {"name": "Wall_0004_StepMiddle", "label": "wall", "center": (5.0, -0.95, 0.9), "size": (3.06, 0.16, 1.8)},
    {"name": "Wall_0005_RightUpper", "label": "wall", "center": (6.25, 1.35, 0.9), "size": (0.16, 4.93, 1.8)},
    {"name": "Wall_0006_TopRight", "label": "wall", "center": (5.15, 3.45, 0.9), "size": (2.89, 0.16, 1.8)},
    {"name": "Wall_0007_StepJoin", "label": "wall", "center": (3.75, 2.65, 0.9), "size": (0.16, 2.125, 1.8)},
    {"name": "Obstacle_0000_LeftIsland", "label": "box", "center": (-2.3, -0.85, 0.9), "size": (0.425, 1.275, 1.8)},
    {"name": "Obstacle_0001_CenterIsland", "label": "box", "center": (0.4, -0.95, 0.9), "size": (0.425, 1.375, 1.8)},
    {"name": "Obstacle_0002_RightIsland", "label": "box", "center": (2.95, -0.65, 0.9), "size": (0.45, 1.4, 1.8)},
)

# Keep the existing actor start points stable so the rest of the surveillance
# pipeline keeps its proven motion envelopes on the new terrain.
_ACTOR_SPAWN_POINTS = {
    "dog1": (-2.0, -2.0, 0.42),
    "dog2": (-2.0, 1.6, 0.42),
    "suspect": (2.0, -0.5, 1.34),
}

# Conservative walkable footprint inside the stepped terrain walls. Keeping the
# polygon slightly inset reduces wall-skimming trajectories and avoids the
# "through-wall" shortcuts that were inflating localization error.
_WALKABLE_POLYGON = (
    (-5.18, -4.36),
    (3.48, -4.36),
    (3.48, -1.14),
    (5.98, -1.14),
    (5.98, 1.34),
    (3.48, 1.34),
    (3.48, 3.24),
    (-5.18, 3.24),
)

# Six surveillance mounts placed on the actual terrain corners of the stepped
# layout. Each camera is pulled inward along a route-facing direction so the
# monocular ground-plane projection works over a shorter, less biased range.
_CAMERA_CORNER_SPECS = (
    {"name": "cam_nw", "corner": (-5.50, 3.45), "look_hint": (-2.20, 1.90), "mount_inset": 1.05},
    {"name": "cam_ne", "corner": (6.25, 3.45), "look_hint": (2.35, 0.85), "mount_inset": 1.10},
    {"name": "cam_e_upper", "corner": (6.25, -0.95), "look_hint": (2.55, 0.55), "mount_inset": 0.95},
    {"name": "cam_e_lower", "corner": (3.75, -0.95), "look_hint": (2.35, -0.65), "mount_inset": 0.85},
    {"name": "cam_se", "corner": (3.75, -4.65), "look_hint": (2.10, -1.45), "mount_inset": 0.95},
    {"name": "cam_sw", "corner": (-5.50, -4.65), "look_hint": (-2.30, -2.35), "mount_inset": 1.00},
)


def _move_toward(source_xy: tuple[float, float], target_xy: tuple[float, float], distance: float) -> tuple[float, float]:
    dx = target_xy[0] - source_xy[0]
    dy = target_xy[1] - source_xy[1]
    norm = math.hypot(dx, dy)
    if norm < 1.0e-6:
        return source_xy
    scale = distance / norm
    return (source_xy[0] + dx * scale, source_xy[1] + dy * scale)


def get_floor_cuboid() -> dict:
    """Return the terrain floor cuboid."""
    return dict(_STATIC_CUBOIDS[0])


def get_map_bounds() -> dict[str, float]:
    """Return axis-aligned floor bounds for the stepped SLAM scene."""
    floor = get_floor_cuboid()
    cx, cy, _ = floor["center"]
    sx, sy, _ = floor["size"]
    half_x = 0.5 * sx
    half_y = 0.5 * sy
    return {
        "x_min": cx - half_x,
        "x_max": cx + half_x,
        "y_min": cy - half_y,
        "y_max": cy + half_y,
        "center_x": cx,
        "center_y": cy,
        "width": sx,
        "height": sy,
    }


def get_static_cuboids(include_floor: bool = False) -> list[dict]:
    """Return all static cuboids for the provided terrain."""
    cuboids = []
    for cuboid in _STATIC_CUBOIDS:
        if cuboid["label"] == "floor" and not include_floor:
            continue
        cuboids.append(dict(cuboid))
    return cuboids


def get_static_cylinders() -> list[dict]:
    """Return cylindrical statics. This terrain uses cuboids only."""
    return []


def get_static_object_positions(labels: tuple[str, ...] = ("wall", "box")) -> list[tuple[float, float, float]]:
    """Return approximate static-object centers for LiDAR clutter filtering."""
    positions = []
    for cuboid in get_static_cuboids(include_floor=False):
        if cuboid["label"] in labels:
            positions.append(tuple(cuboid["center"]))
    return positions


def get_walkable_polygon() -> tuple[tuple[float, float], ...]:
    """Return a conservative polygon for actor path planning."""
    return tuple(_WALKABLE_POLYGON)


def get_blocking_rectangles(
    labels: tuple[str, ...] = ("wall", "box"),
    inflate: float = 0.0,
) -> list[tuple[float, float, float, float]]:
    """Return axis-aligned blocking rectangles as (x_min, x_max, y_min, y_max)."""
    rectangles = []
    margin = max(0.0, float(inflate))
    for cuboid in get_static_cuboids(include_floor=False):
        if cuboid["label"] not in labels:
            continue
        cx, cy, _ = cuboid["center"]
        sx, sy, _ = cuboid["size"]
        hx = 0.5 * sx + margin
        hy = 0.5 * sy + margin
        rectangles.append((cx - hx, cx + hx, cy - hy, cy + hy))
    return rectangles


def _point_in_polygon(x: float, y: float, polygon: tuple[tuple[float, float], ...]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        denom = yj - yi
        if abs(denom) < 1.0e-9:
            denom = 1.0e-9 if denom >= 0.0 else -1.0e-9
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / denom + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def is_point_walkable(x: float, y: float, clearance: float = 0.0) -> bool:
    """Check whether a point lies in the traversable footprint."""
    polygon = get_walkable_polygon()
    if not _point_in_polygon(x, y, polygon):
        return False
    for x_min, x_max, y_min, y_max in get_blocking_rectangles(inflate=clearance):
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return False
    return True


def segment_is_walkable(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    clearance: float = 0.0,
    sample_spacing: float = 0.08,
) -> bool:
    """Check that a line segment stays inside traversable space."""
    x0, y0 = start_xy
    x1, y1 = end_xy
    dist = math.hypot(x1 - x0, y1 - y0)
    num_samples = max(2, int(math.ceil(dist / max(sample_spacing, 1.0e-3))) + 1)
    for idx in range(num_samples):
        alpha = idx / max(num_samples - 1, 1)
        x = x0 + alpha * (x1 - x0)
        y = y0 + alpha * (y1 - y0)
        if not is_point_walkable(x, y, clearance=clearance):
            return False
    return True


def _world_to_grid(
    point_xy: tuple[float, float],
    x_min: float,
    y_min: float,
    resolution: float,
) -> tuple[int, int]:
    return (
        int(round((point_xy[0] - x_min) / resolution)),
        int(round((point_xy[1] - y_min) / resolution)),
    )


def _grid_to_world(
    cell_xy: tuple[int, int],
    x_min: float,
    y_min: float,
    resolution: float,
) -> tuple[float, float]:
    return (
        x_min + cell_xy[0] * resolution,
        y_min + cell_xy[1] * resolution,
    )


def _nearest_free_cell(
    start_cell: tuple[int, int],
    occupancy: list[list[bool]],
) -> tuple[int, int]:
    width = len(occupancy[0])
    height = len(occupancy)
    sx, sy = start_cell
    sx = min(max(sx, 0), width - 1)
    sy = min(max(sy, 0), height - 1)
    if not occupancy[sy][sx]:
        return (sx, sy)

    visited = {(sx, sy)}
    queue = [(sx, sy)]
    for cell in queue:
        cx, cy = cell
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = cx + dx
            ny = cy + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if (nx, ny) in visited:
                continue
            if not occupancy[ny][nx]:
                return (nx, ny)
            visited.add((nx, ny))
            queue.append((nx, ny))
    return (sx, sy)


def _astar_grid_path(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    occupancy: list[list[bool]],
    x_min: float,
    y_min: float,
    resolution: float,
) -> list[tuple[float, float]]:
    width = len(occupancy[0])
    height = len(occupancy)
    start = _nearest_free_cell(_world_to_grid(start_xy, x_min, y_min, resolution), occupancy)
    goal = _nearest_free_cell(_world_to_grid(goal_xy, x_min, y_min, resolution), occupancy)

    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, 0.0, start))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {start: 0.0}
    visited: set[tuple[int, int]] = set()

    neighbors = (
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),            (1, 0),
        (-1, 1),  (0, 1),   (1, 1),
    )

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        if current == goal:
            break
        visited.add(current)

        cx, cy = current
        for dx, dy in neighbors:
            nx = cx + dx
            ny = cy + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if occupancy[ny][nx]:
                continue
            step_cost = math.sqrt(2.0) if dx and dy else 1.0
            tentative_g = current_g + step_cost
            neighbor = (nx, ny)
            if tentative_g >= g_score.get(neighbor, float("inf")):
                continue
            g_score[neighbor] = tentative_g
            came_from[neighbor] = current
            heuristic = math.hypot(goal[0] - nx, goal[1] - ny)
            heapq.heappush(open_heap, (tentative_g + heuristic, tentative_g, neighbor))

    if goal not in came_from and goal != start:
        return [start_xy, goal_xy]

    path_cells = [goal]
    while path_cells[-1] != start:
        path_cells.append(came_from[path_cells[-1]])
    path_cells.reverse()
    return [_grid_to_world(cell, x_min, y_min, resolution) for cell in path_cells]


def _simplify_polyline(
    points_xy: list[tuple[float, float]],
    clearance: float,
) -> list[tuple[float, float]]:
    if len(points_xy) <= 2:
        return points_xy

    simplified = [points_xy[0]]
    anchor_idx = 0
    while anchor_idx < len(points_xy) - 1:
        next_idx = anchor_idx + 1
        for candidate_idx in range(len(points_xy) - 1, anchor_idx, -1):
            if segment_is_walkable(
                points_xy[anchor_idx],
                points_xy[candidate_idx],
                clearance=clearance,
            ):
                next_idx = candidate_idx
                break
        simplified.append(points_xy[next_idx])
        anchor_idx = next_idx
    return simplified


def plan_path_through_waypoints(
    waypoints_xyz: list[tuple[float, float, float]],
    clearance: float = 0.30,
    resolution: float = 0.14,
) -> list[tuple[float, float, float]]:
    """Plan a collision-free polyline through ordered waypoints."""
    if len(waypoints_xyz) <= 1:
        return list(waypoints_xyz)

    bounds = get_map_bounds()
    x_min = bounds["x_min"]
    x_max = bounds["x_max"]
    y_min = bounds["y_min"]
    y_max = bounds["y_max"]

    width = int(math.ceil((x_max - x_min) / resolution)) + 1
    height = int(math.ceil((y_max - y_min) / resolution)) + 1
    occupancy = [[False for _ in range(width)] for _ in range(height)]
    for gy in range(height):
        for gx in range(width):
            wx, wy = _grid_to_world((gx, gy), x_min, y_min, resolution)
            occupancy[gy][gx] = not is_point_walkable(wx, wy, clearance=clearance)

    closed_loop = False
    if len(waypoints_xyz) >= 2:
        start_xy = waypoints_xyz[0][:2]
        end_xy = waypoints_xyz[-1][:2]
        closed_loop = math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]) < max(resolution, 1.0e-6)

    combined_xy: list[tuple[float, float]] = []
    z_ref = float(waypoints_xyz[0][2])
    for start, goal in zip(waypoints_xyz[:-1], waypoints_xyz[1:]):
        segment_xy = _astar_grid_path(
            (float(start[0]), float(start[1])),
            (float(goal[0]), float(goal[1])),
            occupancy,
            x_min,
            y_min,
            resolution,
        )
        if combined_xy:
            segment_xy = segment_xy[1:]
        combined_xy.extend(segment_xy)

    if closed_loop:
        if len(combined_xy) > 1:
            combined_xy = combined_xy[:-1]
        deduped_xy: list[tuple[float, float]] = []
        for point in combined_xy:
            if not deduped_xy or math.hypot(point[0] - deduped_xy[-1][0], point[1] - deduped_xy[-1][1]) > 1.0e-6:
                deduped_xy.append(point)
        combined_xy = deduped_xy
        if combined_xy:
            combined_xy.append(combined_xy[0])
    else:
        combined_xy = _simplify_polyline(combined_xy, clearance=clearance)
    return [(x, y, z_ref) for x, y in combined_xy]


def get_visualization_obstacles() -> list[tuple[float, float, str]]:
    """Return obstacle markers for the top-down visualization."""
    obstacles = []
    for cuboid in get_static_cuboids(include_floor=False):
        if cuboid["label"] == "box":
            cx, cy, _ = cuboid["center"]
            obstacles.append((cx, cy, "box"))
    return obstacles


def get_actor_spawn_points() -> dict[str, tuple[float, float, float]]:
    """Return default actor spawn points."""
    return dict(_ACTOR_SPAWN_POINTS)


def get_surveillance_camera_names() -> tuple[str, ...]:
    """Return the ordered fixed-CCTV names."""
    return tuple(spec["name"] for spec in _CAMERA_CORNER_SPECS)


def get_camera_positions(height: float = 2.35, inset: float = 0.18) -> dict[str, tuple[float, float, float]]:
    """Return CCTV positions pulled inward from the stepped terrain corners."""
    positions = {}
    for spec in _CAMERA_CORNER_SPECS:
        mount_inset = float(spec.get("mount_inset", inset))
        px, py = _move_toward(spec["corner"], spec["look_hint"], mount_inset)
        positions[spec["name"]] = (px, py, height)
    return positions


def get_camera_targets(target_height: float = 0.85) -> dict[str, tuple[float, float, float]]:
    """Return interior aim points for the fixed CCTV cameras."""
    targets = {}
    for spec in _CAMERA_CORNER_SPECS:
        targets[spec["name"]] = (spec["look_hint"][0], spec["look_hint"][1], target_height)
    return targets


def get_camera_markers(height: float = 2.35, inset: float = 0.18) -> list[tuple[float, float, str]]:
    """Return camera marker positions for the dashboard map."""
    markers = []
    for name, position in get_camera_positions(height=height, inset=inset).items():
        markers.append((position[0], position[1], name))
    return markers


MAP_BOUNDS = get_map_bounds()
SCENE_SIZE = max(MAP_BOUNDS["width"], MAP_BOUNDS["height"])
WALL_HEIGHT = max((cuboid["size"][2] for cuboid in get_static_cuboids() if cuboid["label"] == "wall"), default=1.8)
WALL_THICKNESS = min(
    (min(cuboid["size"][0], cuboid["size"][1]) for cuboid in get_static_cuboids() if cuboid["label"] == "wall"),
    default=0.16,
)
PILLAR_RADIUS = 0.0
PILLAR_HEIGHT = 0.0
BOX_SIZES = [tuple(cuboid["size"]) for cuboid in get_static_cuboids() if cuboid["label"] == "box"]
CAM_HEIGHT = 2.35
