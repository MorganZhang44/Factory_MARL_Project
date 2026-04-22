"""
test_env.py
Smoke-test for PursuitEnv:
  1. Build obstacle map + occupancy grid
  2. Run one episode with random actions → check shapes and values
  3. Run A* query → check path is returned
  4. Check capture termination logic

Run: python tests/test_env.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from marl.envs.pursuit_env import PursuitEnv
from isaac.scenes.slam_scene_cfg import ObstacleMap
from marl.utils.astar import astar


CFG_PATH = Path("configs/mappo_config.yaml")


def load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


# ── Test 1: Obstacle map ───────────────────────────────────────────────

def test_obstacle_map():
    cfg = load_cfg()
    om  = ObstacleMap(
        map_half=cfg["env"]["map_half"],
        resolution=cfg["env"]["grid_resolution"],
        agent_radius=cfg["env"]["agent_radius"],
    )
    grid = om.get_grid()
    gs   = int(2 * cfg["env"]["map_half"] / cfg["env"]["grid_resolution"])
    assert grid.shape == (gs, gs), f"Grid shape mismatch: {grid.shape}"
    assert grid.max() == 1, "No obstacles found – check slam_scene_cfg"
    # Centre of map should be free
    rc = om.world_to_grid(0.0, 0.0)
    assert grid[rc] == 0, "Map centre should be obstacle-free"
    # Wall boundary should be occupied
    assert om.is_collision(11.0, 0.0), "Outside boundary should collide"
    print(f"  [PASS] obstacle map  shape={grid.shape}  occupied={int(grid.sum())} cells")


# ── Test 2: A* path ────────────────────────────────────────────────────

def test_astar():
    cfg = load_cfg()
    om  = ObstacleMap(
        map_half=cfg["env"]["map_half"],
        resolution=cfg["env"]["grid_resolution"],
        agent_radius=cfg["env"]["agent_radius"],
    )
    # Dog-1 start → suspect start (open space, should find path)
    path = astar(om, (3.0, 3.0), (5.0, 0.0))
    assert isinstance(path, list) and len(path) > 0, "A* returned empty path"
    # All waypoints should be within bounds
    for (x, y) in path:
        assert -10.5 <= x <= 10.5 and -10.5 <= y <= 10.5, f"Waypoint out of bounds: ({x},{y})"
    print(f"  [PASS] A* path  waypoints={len(path)}")


# ── Test 3: Environment step ───────────────────────────────────────────

def test_env_step():
    cfg = load_cfg()
    env = PursuitEnv(cfg, render_mode=None)

    obs, info = env.reset()
    obs_dim = cfg["env"].get("obs_dim", 13)
    assert obs.shape == (cfg["env"]["n_agents"], obs_dim), f"Obs shape wrong: {obs.shape}"

    total_r = 0.0
    for step in range(10):
        action = env.action_space.sample()
        obs, rewards, terminated, truncated, info = env.step(action)
        assert obs.shape == (cfg["env"]["n_agents"], obs_dim)
        assert rewards.shape == (cfg["env"]["n_agents"],)
        assert isinstance(terminated, bool)
        total_r += float(rewards.mean())

    env.close()
    print(f"  [PASS] env step  10 steps completed  avg_reward={total_r/10:.3f}")


# ── Test 4: Capture termination ────────────────────────────────────────

def test_capture():
    cfg = load_cfg()
    env = PursuitEnv(cfg, render_mode=None)
    env.reset()

    # Force agent right on top of suspect
    env.agent_pos[0] = env.target_pos.copy()
    action = np.zeros((cfg["env"]["n_agents"], 2))
    _, _, terminated, _, info = env.step(action)

    assert info["captured"], "Capture not detected when agent overlaps suspect"
    env.close()
    print("  [PASS] capture termination")


# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running environment tests...\n")
    test_obstacle_map()
    test_astar()
    test_env_step()
    test_capture()
    print("\nAll tests passed ✓")
