"""
pursuit_env.py
2D multi-agent pursuit environment (Gymnasium-compatible).

Mirrors the Isaac Lab warehouse scene (warehouse_scene_cfg.py):
  - 20 x 20 m map, origin at (0, 0), range [-10, +10]
  - 2 Unitree Go2 agents
  - 1 Humanoid suspect (controllable by an intruder policy or scripted random walk)

MARL interface (per step):
  obs    : np.ndarray  (n_agents, 21 or 22)  # 21 if use_roles=False, 22 if True
  action : np.ndarray  (n_agents, 2)         ← relative subgoal offset [dx, dy]
  reward : np.ndarray  (n_agents,)           ← per-agent reward (per-role when use_roles=True)
  terminated/truncated: bool
  info   : dict — when an intruder policy is wired in, also contains
           "intruder_obs" (20,)  — next-state observation for the intruder policy
           "intruder_reward" (float) — reward for the intruder action just executed

Roles (when use_roles=True):
  PURSUER (0)   : closer to target — gets dense closing reward
  ENCIRCLER (1) : farther — gets angular encirclement reward
  Re-evaluated each step but only switched when:
    steps_since_switch >= role_cooldown_steps  AND  |d0 - d1| >= role_gap_threshold
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple

from ..utils.map_utils import ObstacleMap
from ..utils.astar import astar
from ..rewards.pursuit_reward import PursuitReward, RolePursuitReward, IntruderReward


PURSUER = 0
ENCIRCLER = 1


class PursuitEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, cfg: dict, render_mode: Optional[str] = None):
        super().__init__()

        env_cfg = cfg["env"]
        self.map_half           = float(env_cfg["map_half"])
        self.n_agents           = int(env_cfg["n_agents"])
        self.max_steps          = int(env_cfg["max_steps"])
        self.dt                 = float(env_cfg["dt"])
        self.capture_radius     = float(env_cfg["capture_radius"])
        self.agent_max_spd      = float(env_cfg["agent_max_speed"])
        self.intruder_spd       = float(env_cfg["intruder_speed"])
        self.intruder_max_off   = float(env_cfg.get("intruder_max_offset", 2.0))
        self.agent_max_off      = float(env_cfg.get("agent_max_offset", 3.0))
        self.agent_radius       = float(env_cfg["agent_radius"])
        self.grid_res           = float(env_cfg["grid_resolution"])
        self.render_mode        = render_mode

        # Role mechanism config
        self.use_roles          = bool(env_cfg.get("use_roles", False))
        self.role_cooldown      = int(env_cfg.get("role_cooldown_steps", 30))
        self.role_gap_thresh    = float(env_cfg.get("role_gap_threshold", 1.0))
        # Probability that the intruder spawns NEAR a wall/obstacle (≤1 m).
        # This trains the policy on the failure mode where users park the
        # intruder against a wall in the demo.
        self.near_obstacle_prob = float(env_cfg.get("intruder_near_obstacle_prob", 0.0))
        # Probability that BOTH agents spawn on the same side of the intruder
        # (within a 90° wedge). Forces the policy to learn the recovery action
        # of one agent splitting off to flank, matching the demo failure case.
        self.same_side_spawn_prob = float(env_cfg.get("same_side_spawn_prob", 0.0))
        # Intruder vision: how many lidar rays and range. Higher resolution lets
        # the evader spot escape routes in detail, useful for adversarial training.
        self.intruder_lidar_rays  = int(env_cfg.get("intruder_lidar_rays", 8))
        self.intruder_lidar_range = float(env_cfg.get("intruder_lidar_range", 8.0))

        # Obstacle map
        self.obs_map = ObstacleMap(
            map_half=self.map_half,
            resolution=self.grid_res,
            agent_radius=self.agent_radius,
        )

        # Reward function — pick role-aware variant if requested
        if self.use_roles:
            self.reward_fn = RolePursuitReward(cfg["reward"])
        else:
            self.reward_fn = PursuitReward(cfg["reward"])
        self.intruder_reward_fn = IntruderReward(cfg["reward"])

        # Gym spaces — obs_dim must match cfg (21 base, 22 with role flag)
        obs_dim = int(env_cfg.get("obs_dim", 22 if self.use_roles else 21))
        self.obs_dim = obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_agents, obs_dim), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-5.0, high=5.0,
            shape=(self.n_agents, 2), dtype=np.float32,
        )
        # Intruder obs dim: 12 base state + 8 lidar = 20
        self.intruder_obs_dim = int(env_cfg.get("intruder_obs_dim", 20))

        # Episode state
        self.agent_pos  = np.zeros((self.n_agents, 2), dtype=np.float64)
        self.agent_vel  = np.zeros((self.n_agents, 2), dtype=np.float64)
        self.target_pos = np.zeros(2, dtype=np.float64)
        self.target_vel = np.zeros(2, dtype=np.float64)
        self._subgoals  = np.zeros((self.n_agents, 2), dtype=np.float64)
        self._paths: List[List[Tuple[float, float]]] = [[] for _ in range(self.n_agents)]
        self._step_count = 0

        # Role state
        self.roles                = np.zeros(self.n_agents, dtype=np.int64)
        self.steps_since_switch   = 0
        self._intruder_hit_wall   = False
        # Anti-stuck tracking: how many consecutive steps both agents have been
        # inside capture_radius without satisfying the 90° capture condition
        self._stuck_steps         = 0

        # Rendering handles
        self._fig = None
        self._ax  = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._intruder_hit_wall = False
        self._stuck_steps = 0

        safe_r = self.map_half - 1.5

        # Decide spawn topology
        same_side = (self.same_side_spawn_prob > 0.0
                     and self.np_random.random() < self.same_side_spawn_prob)

        # First spawn the intruder so we have a reference point for same-side
        place_near = (self.near_obstacle_prob > 0.0
                      and self.np_random.random() < self.near_obstacle_prob)
        tpos = None
        if place_near:
            tpos = self._sample_near_obstacle(safe_r, min_dist_from_agents=0.0)
        if tpos is None:
            while True:
                cand = self.np_random.uniform(-safe_r, safe_r, 2)
                if not self.obs_map.is_collision(cand[0], cand[1]):
                    tpos = cand
                    break
        self.target_pos = tpos

        # Now spawn agents
        if same_side:
            # Pick a random direction; place both agents in a 90° wedge from
            # target, at random distances 2.5–6.0 m
            base_angle = self.np_random.uniform(0, 2 * np.pi)
            for i in range(self.n_agents):
                ok = False
                for _ in range(50):
                    ang = base_angle + self.np_random.uniform(-np.pi / 4, np.pi / 4)
                    dist = self.np_random.uniform(2.5, 6.0)
                    pos = tpos + dist * np.array([np.cos(ang), np.sin(ang)])
                    pos = np.clip(pos, -safe_r, safe_r)
                    if not self.obs_map.is_collision(pos[0], pos[1]):
                        if i == 0 or np.linalg.norm(pos - self.agent_pos[0]) > 0.6:
                            self.agent_pos[i] = pos
                            ok = True
                            break
                if not ok:
                    # Fallback to random valid position
                    while True:
                        pos = self.np_random.uniform(-safe_r, safe_r, 2)
                        if not self.obs_map.is_collision(pos[0], pos[1]):
                            self.agent_pos[i] = pos
                            break
        else:
            # Original random spawn (>3 m from target, >2 m from teammate)
            for i in range(self.n_agents):
                while True:
                    pos = self.np_random.uniform(-safe_r, safe_r, 2)
                    if self.obs_map.is_collision(pos[0], pos[1]):
                        continue
                    if np.linalg.norm(pos - tpos) <= 3.0:
                        continue
                    if i == 0 or np.linalg.norm(pos - self.agent_pos[0]) > 2.0:
                        self.agent_pos[i] = pos
                        break

        self.agent_vel[:] = 0.0
        angle = self.np_random.uniform(0, 2 * np.pi)
        self.target_vel = np.array([np.cos(angle), np.sin(angle)]) * self.intruder_spd

        self._subgoals = self.agent_pos.copy()
        self._paths    = [[] for _ in range(self.n_agents)]

        # Initial role assignment: closer agent = PURSUER
        if self.use_roles and self.n_agents == 2:
            d = np.linalg.norm(self.agent_pos - self.target_pos, axis=1)
            if d[0] <= d[1]:
                self.roles[0], self.roles[1] = PURSUER, ENCIRCLER
            else:
                self.roles[0], self.roles[1] = ENCIRCLER, PURSUER
        else:
            self.roles[:] = PURSUER
        self.steps_since_switch = 0

        return self._get_obs(), {}

    def step(self, actions: np.ndarray, intruder_action: Optional[np.ndarray] = None):
        """
        actions          : (n_agents, 2) — relative offset [dx, dy] per agent
        intruder_action  : (2,) optional — relative offset for intruder; if None,
                           env falls back to scripted random walk
        """
        self._step_count += 1
        self._intruder_hit_wall = False
        dists_before = np.linalg.norm(self.agent_pos - self.target_pos, axis=1)
        target_pos_before = self.target_pos.copy()

        # Convert relative agent action → subgoal (norm-clip so direction is preserved)
        for i in range(self.n_agents):
            raw = np.array(actions[i], dtype=np.float64)
            mag = float(np.linalg.norm(raw))
            if mag > self.agent_max_off:
                raw = (raw / mag) * self.agent_max_off
            sg_world = self.agent_pos[i] + raw
            sg_world = np.clip(sg_world, -self.map_half + 0.3, self.map_half - 0.3)
            self._subgoals[i] = sg_world

        # 1. Move each agent via A*
        for i in range(self.n_agents):
            sg = tuple(self._subgoals[i])
            self._paths[i] = astar(self.obs_map, tuple(self.agent_pos[i]), sg)
            # Sequential movement with agent-agent collision: agent i must not
            # overlap with the OTHER agent's *current* position.
            other_pos = None
            if self.n_agents == 2:
                other_pos = self.agent_pos[1 - i].copy()
            self.agent_pos[i], self.agent_vel[i] = self._move_along_path(
                self.agent_pos[i], self._paths[i], self.agent_max_spd,
                other_pos=other_pos,
            )

        # 2. Step intruder — RL action if provided, otherwise scripted
        if intruder_action is not None:
            self.target_pos, self.target_vel = self._step_intruder_rl(intruder_action)
        else:
            self.target_pos, self.target_vel = self._step_intruder()

        # 3. Distances after step
        dists_after = np.linalg.norm(self.agent_pos - self.target_pos, axis=1)

        # 4. Re-evaluate roles with hysteresis (still computed on before-step snapshot)
        if self.use_roles and self.n_agents == 2:
            self._maybe_switch_roles(dists_after)

        # 5. Termination check (computed BEFORE reward so anti-stuck tracking works)
        captured = False
        both_in_capture = False
        if self.n_agents >= 2:
            d1, d2 = float(dists_after[0]), float(dists_after[1])
            both_in_capture = (d1 <= self.capture_radius and d2 <= self.capture_radius)
            if both_in_capture:
                v1 = self.agent_pos[0] - self.target_pos
                v2 = self.agent_pos[1] - self.target_pos
                cos_theta = float(np.dot(v1, v2) / (d1 * d2 + 1e-8))
                if cos_theta <= 0.0:
                    captured = True
        else:
            captured = bool(np.any(dists_after <= self.capture_radius))

        # 6. Anti-stuck counter: increment when "both inside capture range but no capture"
        #    Reset on either successful capture or one agent leaving the zone.
        if both_in_capture and not captured:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0

        # 7. Reward (with stuck_steps so reward fn can apply anti-stuck penalty)
        if self.use_roles:
            rewards = self.reward_fn.compute_per_agent(
                agent_pos=self.agent_pos,
                target_pos=self.target_pos,
                target_vel=self.target_vel,
                roles=self.roles,
                capture_radius=self.capture_radius,
                dists_before=dists_before,
                dists_after=dists_after,
                step_count=self._step_count,
                max_steps=self.max_steps,
                stuck_steps=self._stuck_steps,
            )
        else:
            rewards = self.reward_fn.compute(
                self.agent_pos, self.target_pos, self.capture_radius,
                dists_before=dists_before, dists_after=dists_after,
                step_count=self._step_count, max_steps=self.max_steps,
            )

        terminated = captured
        truncated  = self._step_count >= self.max_steps

        # 7. Intruder reward (always computed; trainer may ignore if no intruder policy)
        intruder_min_dist_before = float(np.min(dists_before))
        intruder_min_dist_after  = float(np.min(dists_after))
        intruder_reward = self.intruder_reward_fn.compute(
            min_dist_before=intruder_min_dist_before,
            min_dist_after=intruder_min_dist_after,
            captured=captured,
            hit_wall=self._intruder_hit_wall,
        )

        info = {
            "captured": captured,
            "min_dist": intruder_min_dist_after,
            "step": self._step_count,
            "roles": self.roles.copy(),
            "intruder_reward": float(intruder_reward),
            "intruder_obs": self._get_intruder_obs(),
            "intruder_terminated": captured,
            "target_displacement": float(np.linalg.norm(self.target_pos - target_pos_before)),
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    # Spawn helpers
    # ------------------------------------------------------------------

    def _sample_near_obstacle(
        self,
        safe_r: float,
        min_dist_from_agents: float = 3.0,
        wall_dist: float = 1.0,
        max_tries: int = 100,
    ) -> Optional[np.ndarray]:
        """Sample a free position with at least one wall/obstacle within `wall_dist` m.
        Returns None if no such position is found in `max_tries` random samples."""
        for _ in range(max_tries):
            cand = self.np_random.uniform(-safe_r, safe_r, 2)
            if self.obs_map.is_collision(float(cand[0]), float(cand[1])):
                continue
            if np.any(np.linalg.norm(self.agent_pos - cand, axis=1) < min_dist_from_agents):
                continue
            # Probe 8 directions × 3 distances for a wall hit
            near = False
            for d in (0.4, 0.7, 1.0):
                for k in range(8):
                    theta = k * np.pi / 4
                    probe = cand + d * np.array([np.cos(theta), np.sin(theta)])
                    if self.obs_map.is_collision(float(probe[0]), float(probe[1])):
                        near = True
                        break
                if near:
                    break
            if near:
                return cand
        return None

    # ------------------------------------------------------------------
    # Role switching with hysteresis
    # ------------------------------------------------------------------

    def _maybe_switch_roles(self, dists_after: np.ndarray):
        """Re-assign roles if cooldown has elapsed AND distance gap is significant."""
        self.steps_since_switch += 1
        if self.steps_since_switch < self.role_cooldown:
            return
        gap = abs(float(dists_after[0]) - float(dists_after[1]))
        if gap < self.role_gap_thresh:
            return
        # Closer agent should be PURSUER
        desired = np.zeros(self.n_agents, dtype=np.int64)
        if dists_after[0] <= dists_after[1]:
            desired[0], desired[1] = PURSUER, ENCIRCLER
        else:
            desired[0], desired[1] = ENCIRCLER, PURSUER
        if not np.array_equal(desired, self.roles):
            self.roles = desired
            self.steps_since_switch = 0

    # ------------------------------------------------------------------
    # Movement helpers
    # ------------------------------------------------------------------

    def _move_along_path(
        self,
        pos: np.ndarray,
        path: List[Tuple[float, float]],
        max_spd: float,
        other_pos: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Move one agent along its A* path.

        If `other_pos` is given, the move is also rejected when it would put
        the agent within (2·agent_radius + 0.1) m of that point — i.e. the
        two physical bodies overlap. Sliding (axis-by-axis) is attempted
        before giving up. This keeps a hard separation of ≈ 0.7 m between
        the two dogs at all times.
        """
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

        min_sep = 2.0 * self.agent_radius + 0.1   # 0.7 m

        def _hits_other(p: np.ndarray) -> bool:
            return (other_pos is not None
                    and float(np.linalg.norm(p - other_pos)) < min_sep)

        def _blocked(p: np.ndarray) -> bool:
            return self.obs_map.is_collision(float(p[0]), float(p[1])) or _hits_other(p)

        # Wall + agent-agent sliding
        if _blocked(new_pos):
            slide_x = np.array([new_pos[0], pos[1]])
            if not _blocked(slide_x):
                return slide_x, np.array([vel[0], 0.0])
            slide_y = np.array([pos[0], new_pos[1]])
            if not _blocked(slide_y):
                return slide_y, np.array([0.0, vel[1]])
            return pos.copy(), np.zeros(2)

        return new_pos, vel

    def _step_intruder_rl(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Execute an RL-provided relative offset for the intruder.
        Same A*-style move-along-path mechanic as the agents, but with intruder_max_off and intruder_spd.
        """
        raw = np.array(action, dtype=np.float64)
        mag = float(np.linalg.norm(raw))
        if mag > self.intruder_max_off:
            raw = (raw / mag) * self.intruder_max_off
        sg = self.target_pos + raw
        sg = np.clip(sg, -self.map_half + 0.3, self.map_half - 0.3)

        path = astar(self.obs_map, tuple(self.target_pos), tuple(sg))
        new_pos, new_vel = self._move_along_path(self.target_pos, path, self.intruder_spd)

        # If we couldn't move (fully blocked), flag wall hit
        if np.allclose(new_pos, self.target_pos) and mag > 1e-3:
            self._intruder_hit_wall = True

        return new_pos, new_vel

    def _step_intruder(self) -> Tuple[np.ndarray, np.ndarray]:
        """Random walk + 20% stationary + pinned-when-cornered. Used as scripted fallback."""
        dists = np.linalg.norm(self.agent_pos - self.target_pos, axis=1)
        if np.any(dists < self.capture_radius):
            return self.target_pos.copy(), np.zeros(2)

        # Stationary 40% of the time (was 20%) — train policy more on the
        # "intruder doesn't move" case which appears when user holds mouse still.
        if self.np_random.random() < 0.40:
            return self.target_pos.copy(), np.zeros(2)

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
            self._intruder_hit_wall = True
            vel     = -vel
            new_pos = self.target_pos + vel * self.dt
            if self.obs_map.is_collision(float(new_pos[0]), float(new_pos[1])):
                new_pos = self.target_pos.copy()

        new_pos = np.clip(new_pos, -self.map_half + 0.6, self.map_half - 0.6)
        return new_pos, vel

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        pos_scale = self.map_half
        vel_scale = max(self.agent_max_spd, self.intruder_spd)
        lidar_angles = [i * np.pi / 4 for i in range(8)]
        lidar_max    = 8.0

        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        for i in range(self.n_agents):
            j = 1 - i
            px, py = float(self.agent_pos[i][0]), float(self.agent_pos[i][1])
            lidar = np.array([
                self.obs_map.ray_cast(px, py, a, max_range=lidar_max)
                for a in lidar_angles
            ], dtype=np.float32)
            base = [
                np.array([float(i)], dtype=np.float32),
            ]
            if self.use_roles and self.obs_dim >= 22:
                base.append(np.array([float(self.roles[i])], dtype=np.float32))
            base.extend([
                self.agent_pos[i] / pos_scale,
                self.agent_vel[i] / vel_scale,
                self.agent_pos[j] / pos_scale,
                self.agent_vel[j] / vel_scale,
                self.target_pos   / pos_scale,
                self.target_vel   / vel_scale,
                lidar,
            ])
            obs[i] = np.concatenate(base)
        return obs

    def _get_intruder_obs(self) -> np.ndarray:
        """Build the intruder's local observation (20D)."""
        pos_scale = self.map_half
        vel_scale = max(self.agent_max_spd, self.intruder_spd)
        lidar_angles = [i * np.pi / 4 for i in range(8)]
        lidar_max    = 8.0
        tx, ty = float(self.target_pos[0]), float(self.target_pos[1])
        lidar = np.array([
            self.obs_map.ray_cast(tx, ty, a, max_range=lidar_max)
            for a in lidar_angles
        ], dtype=np.float32)
        return np.concatenate([
            self.target_pos / pos_scale,
            self.target_vel / vel_scale,
            self.agent_pos[0] / pos_scale,
            self.agent_vel[0] / vel_scale,
            self.agent_pos[1] / pos_scale,
            self.agent_vel[1] / vel_scale,
            lidar,
        ]).astype(np.float32)

    def get_intruder_obs(self) -> np.ndarray:
        """Public accessor used by the trainer to fetch the intruder obs after reset."""
        return self._get_intruder_obs()

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
        grid = self.obs_map.get_grid()
        ax.imshow(
            np.flipud(grid), cmap="gray_r", alpha=0.4,
            extent=[-self.map_half, self.map_half, -self.map_half, self.map_half],
        )

        # Color by role: PURSUER orange-red, ENCIRCLER blue
        role_colors = ["tomato", "royalblue"]
        for i in range(self.n_agents):
            col = role_colors[int(self.roles[i])] if self.use_roles else "royalblue"
            c = mpatches.Circle(self.agent_pos[i], self.agent_radius, color=col)
            ax.add_patch(c)
            ax.plot(*self._subgoals[i], "x", color=col, ms=8, mew=2)
            if self._paths[i]:
                px, py = zip(*self._paths[i])
                ax.plot(px, py, "--", color=col, alpha=0.45, lw=1)

        ax.plot(*self.target_pos, "k*", ms=15, label="Suspect")
        ax.set_xlim(-self.map_half, self.map_half)
        ax.set_ylim(-self.map_half, self.map_half)
        ax.set_aspect("equal")
        ax.set_title(f"Step {self._step_count}  roles={self.roles.tolist()}")
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
