# map_utils.py — re-exports from the unified scene config.
# All map/obstacle definitions live in isaac/scenes/slam_scene_cfg.py.
from isaac.scenes.slam_scene_cfg import (  # noqa: F401
    RectObstacle, CircleObstacle,
    PERIMETER_WALLS, INTERIOR_WALLS, PILLARS, BOXES,
    ALL_RECTS, ALL_CIRCLES,
    MAP_HALF, WALL_THICKNESS,
    AGENT_1_POS_2D, AGENT_2_POS_2D, INTRUDER_POS_2D,
    ObstacleMap,
)
