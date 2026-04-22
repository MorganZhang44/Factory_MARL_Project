"""
pursuit_env.py
2D multi-agent pursuit environment (Gymnasium-compatible).

Mirrors the Isaac Lab SLAM scene (isaac/scenes/slam_scene_cfg.py):
  - 20 x 20 m map, origin at (0, 0), range [-10, +10]
  - 2 Unitree Go2 agents (random spawn, min 2m apart)
  - 1 Humanoid intruder (H1, random spawn, min 3m from agents)

MARL interface (per step):
  obs    : np.ndarray  (n_agents, 13)  [agent_id + 12-D state]
  action : np.ndarray  (n_agents, 2)   subgoal offset [dx, dy] in metres
  reward : np.ndarray  (n_agents,)     shared team reward
  terminated/truncated: bool
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple

from ..utils.map_utils import ObstacleMap
from ..utils.astar import astar
from ..rewards.pursuit_reward import PursuitReward


class PursuitEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, cfg: dict, render_mode: Optional[str] = None):
        super().__init__()

        env_cfg = cfg["env"]
        self.map_half       = float(env_cfg["map_half"])         # 10.0
        self.n_agents       = int(env_cfg["n_agents"])           # 2
        self.max_steps      = int(env_cfg["max_steps"])          # 500
        self.dt             = float(env_cfg["dt"])               # 0.1
        self.capture_radius = float(env_cfg["capture_radius"])   # 1.0
        self.agent_max_spd  = float(env_cfg["agent_max_speed"])  # 1.5
        self.intruder_spd   = float(env_cfg["intruder_speed"])   # 1.0
        self.agent_radius   = float(env_cfg["agent_radius"])     # 0.3
        self.grid_res       = float(env_cfg["grid_resolution"])  # 0.5
        self.render_mode    = render_mode

        # Obstacle map (built once, reused every episode)
        self.obs_map = ObstacleMap(
            map_half=self.map_half,
            resolution=self.grid_res,
            agent_radius=self.agent_radius,
        )

        # Reward function
        self.reward_fn = PursuitReward(cfg["reward"])

        # Gym spaces
        obs_dim = int(env_cfg.get("obs_dim", 13))  # 12D state + 1D agent_id
        self.obs_dim = obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_agents, obs_dim), dtype=np.float32,
        )
        # FIX ②: Relative action — subgoal offset from agent's current position
        # Range [-5, +5] m is enough to reach anywhere in one or two steps
        self.action_space = spaces.Box(
            low=-5.0, high=5.0,
            shape=(self.n_agents, 2), dtype=np.float32,
        )

        # Episode state (initialised in reset)
        self.agent_pos  = np.zeros((self.n_agents, 2), dtype=np.float64)
        self.agent_vel  = np.zeros((self.n_agents, 2), dtype=np.float64)
        self.target_pos = np.zeros(2, dtype=np.float64)
        self.target_vel = np.zeros(2, dtype=np.float64)
        self._subgoals  = np.zeros((self.n_agents, 2), dtype=np.float64)
        self._paths: List[List[Tuple[float, float]]] = [[] for _ in range(self.n_agents)]
        self._step_count = 0

        # Rendering handles
        self._fig = None
        self._ax  = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        # FIX ④: Random spawn positions (within safe open areas)
        safe_r = self.map_half - 1.5  # stay away from walls
        for i in range(self.n_agents):
            while True:
                pos = self.np_random.uniform(-safe_r, safe_r, 2)
                if not self.obs_map.is_collision(pos[0], pos[1]):
                    # ensure agents don't spawn too close to each other
                    if i == 0 or np.linalg.norm(pos - self.agent_pos[0]) > 2.0:
                        self.agent_pos[i] = pos
                        break
        while True:
            tpos = self.np_random.uniform(-safe_r, safe_r, 2)
            dists = np.linalg.norm(self.agent_pos - tpos, axis=1)
            if (not self.obs_map.is_collision(tpos[0], tpos[1])
                    and np.all(dists > 3.0)):  # start at least 3m from agents
                self.target_pos = tpos
                break

        self.agent_vel[:] = 0.0
        angle = self.np_random.uniform(0, 2 * np.pi)
        self.target_vel = np.array([np.cos(angle), np.sin(angle)]) * self.intruder_spd

        self._subgoals = self.agent_pos.copy()
        self._paths    = [[] for _ in range(self.n_agents)]

        return self._get_obs(), {}

    def step(self, actions: np.ndarray):
        """
        FIX ②: actions are RELATIVE offset from agent position → absolute subgoal.
        actions: (n_agents, 2) — offset [dx, dy] in metres.
        """
        self._step_count += 1
        dists_before = np.linalg.norm(self.agent_pos - self.target_pos, axis=1)

        # Convert relative action → absolute subgoal, clamped to map bounds
        for i in range(self.n_agents):
            offset = np.clip(actions[i], -5.0, 5.0)
            sg_world = self.agent_pos[i] + offset
            sg_world = np.clip(sg_world, -self.map_half + 0.3, self.map_half - 0.3)
            self._subgoals[i] = sg_world

        # 1. Move each agent via A* toward its subgoal
        for i in range(self.n_agents):
            sg = tuple(self._subgoals[i])
            self._paths[i] = astar(self.obs_map, tuple(self.agent_pos[i]), sg)
            self.agent_pos[i], self.agent_vel[i] = self._move_along_path(
                self.agent_pos[i], self._paths[i], self.agent_max_spd
            )

        # 2. Step intruder (random walk)
        self.target_pos, self.target_vel = self._step_intruder()

        # 3. Reward (FIX ③: pass dists_before for dense progress reward)
        dists_after = np.linalg.norm(self.agent_pos - self.target_pos, axis=1)
        rewards = self.reward_fn.compute(
            self.agent_pos, self.target_pos, self.capture_radius,
            dists_before=dists_before, dists_after=dists_after,
        )

        # 4. Termination
        dists     = dists_after
        captured  = bool(np.any(dists <= self.capture_radius))
        terminated = captured
        truncated  = self._step_count >= self.max_steps

        info = {"captured": captured, "min_dist": float(np.min(dists)), "step": self._step_count}

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    # Movement helpers
    # ------------------------------------------------------------------

    def _move_along_path(
        self,
        pos: np.ndarray,
        path: List[Tuple[float, float]],
        max_spd: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not path:
            return pos.copy(), np.zeros(2)

        target = np.array(path[0])
        diff   = target - pos
        dist   = float(np.linalg.norm(diff))

        if dist < 1e-6:
            if len(path) > 1:
                target = np.array(path[1])
                diff   = target - pos
                dist   = float(np.linalg.norm(diff))
            else:
                return pos.copy(), np.zeros(2)

        step   = min(max_spd * self.dt, dist)
        unit   = diff / dist
        vel    = unit * (step / self.dt)
        new_pos = pos + unit * step

        # Collision safety: stay if new position is blocked
        if self.obs_map.is_collision(float(new_pos[0]), float(new_pos[1])):
            return pos.copy(), np.zeros(2)

        return new_pos, vel

    def _step_intruder(self) -> Tuple[np.ndarray, np.ndarray]:
        """Random walk with wall/obstacle reflection."""
        noise = self.np_random.normal(0, 0.3, 2)
        vel   = self.target_vel + noise
        spd   = float(np.linalg.norm(vel))
        if spd > 1e-6:
            vel = vel / spd * self.intruder_spd
        else:
            angle = self.np_random.uniform(0, 2 * np.pi)
            vel   = np.array([np.cos(angle), np.sin(angle)]) * self.intruder_spd

        new_pos = self.target_pos + vel * self.dt
        if self.obs_map.is_collision(float(new_pos[0]), float(new_pos[1])):
            vel     = -vel
            new_pos = self.target_pos + vel * self.dt
            if self.obs_map.is_collision(float(new_pos[0]), float(new_pos[1])):
                new_pos = self.target_pos.copy()

        new_pos = np.clip(new_pos, -self.map_half + 0.6, self.map_half - 0.6)
        return new_pos, vel

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        # Normalize to [-1, +1]
        pos_scale = self.map_half
        vel_scale = max(self.agent_max_spd, self.intruder_spd)
        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        for i in range(self.n_agents):
            j = 1 - i  # teammate
            obs[i] = np.concatenate([
                [float(i)],                          # agent_id (0 or 1) ← KEY for role differentiation
                self.agent_pos[i] / pos_scale,       # self_pos   (2)
                self.agent_vel[i] / vel_scale,       # self_vel   (2)
                self.agent_pos[j] / pos_scale,       # mate_pos   (2)
                self.agent_vel[j] / vel_scale,       # mate_vel   (2)
                self.target_pos   / pos_scale,       # target_pos (2)
                self.target_vel   / vel_scale,       # target_vel (2)
            ])
        return obs

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(7, 7))

        ax = self._ax
        ax.clear()

        # Occupancy grid
        grid = self.obs_map.get_grid()
        ax.imshow(
            np.flipud(grid), cmap="gray_r", alpha=0.4,
            extent=[-self.map_half, self.map_half, -self.map_half, self.map_half],
        )

        # Agents
        colors = ["royalblue", "steelblue"]
        for i in range(self.n_agents):
            c = mpatches.Circle(self.agent_pos[i], self.agent_radius, color=colors[i])
            ax.add_patch(c)
            ax.plot(*self._subgoals[i], "x", color=colors[i], ms=8, mew=2)
            if self._paths[i]:
                px, py = zip(*self._paths[i])
                ax.plot(px, py, "--", color=colors[i], alpha=0.45, lw=1)

        # Suspect
        ax.plot(*self.target_pos, "r*", ms=15, label="Suspect")

        ax.set_xlim(-self.map_half, self.map_half)
        ax.set_ylim(-self.map_half, self.map_half)
        ax.set_aspect("equal")
        ax.set_title(f"Step {self._step_count}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
