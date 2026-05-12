#!/usr/bin/env python3
"""MARL decision module HTTP adapter.

This module owns only the runtime inference boundary for MARL. Training code
stays in the research-oriented files under ``marl/`` and is not part of the
system control path.

Input:
  - robot world-frame position/velocity for both agents
  - intruder world-frame position/velocity

Output:
  - one world-frame subgoal per robot

Default runtime resources are aligned with the in-tree ``v13`` release under
``marl/marl_pursuit_v13_final/release_v13_final``. If that checkpoint is
unavailable, the service falls back to a simple intercept offset so the module
boundary can still be exercised end to end.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
V13_RELEASE_ROOT = MODULE_ROOT / "marl_pursuit_v13_final" / "release_v13_final"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V13_RELEASE_ROOT))
from marl.utils.map_utils import ObstacleMap

V13_ROOT = V13_RELEASE_ROOT
DEFAULT_CHECKPOINT = V13_ROOT / "checkpoints" / "v13_final.pt"
LEGACY_CHECKPOINT = MODULE_ROOT / "checkpoints" / "mappo_latest.pt"
DEFAULT_ROBOT_IDS = ("agent_1", "agent_2")
PURSUER = 0
ENCIRCLER = 1
DEFAULT_ROLE_COOLDOWN_STEPS = 30
DEFAULT_ROLE_GAP_THRESHOLD = 1.0
DEFAULT_GRID_RESOLUTION = 0.5
DEFAULT_AGENT_RADIUS = 0.3
DEFAULT_LIDAR_RANGE = 8.0
DEFAULT_LIDAR_RAYS = 8
DEFAULT_TRACE_DIR = PROJECT_ROOT / "output" / "marl_traces"
DEFAULT_CAPTURE_RADIUS = 1.5
DEFAULT_FACE_CAPTURE_THRESH = 0.5
DEFAULT_AGENT_MAX_OMEGA = 0.6
SG_COMMIT_FRAMES = 10
SG_REACHED_THRESH = 0.5
SG_TARGET_DRIFT = 1.5
SG_FALLBACK_CAP = 6.0
SG_POLICY_CAP = 3.0


def _clip_offset_norm(offset: np.ndarray, limit: float) -> np.ndarray:
    clipped = np.asarray(offset, dtype=np.float32).copy()
    mag = float(np.linalg.norm(clipped))
    if mag > limit > 0.0:
        clipped *= limit / mag
    return clipped


def _as_xy(value: Any, field_name: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) < 2:
        raise ValueError(f"{field_name} must be a 2D coordinate")
    return float(value[0]), float(value[1])


def _clip_world(point: np.ndarray, map_half: float) -> np.ndarray:
    limit = max(0.0, map_half - 0.3)
    return np.clip(point, -limit, limit)


class MarlPolicyRunner:
    def __init__(
        self,
        checkpoint: Path,
        map_half: float,
        vel_scale: float,
        action_limit: float,
        grid_resolution: float,
        agent_radius: float,
        role_cooldown_steps: int,
        role_gap_threshold: float,
        lidar_range: float,
        lidar_rays: int,
        trace_dir: Path | None = None,
        deterministic: bool = True,
        fallback_enabled: bool = True,
        fallback_only: bool = False,
    ) -> None:
        self.checkpoint = checkpoint
        self.map_half = float(map_half)
        self.vel_scale = float(max(vel_scale, 1.0e-6))
        self.action_limit = float(max(action_limit, 1.0e-6))
        self.grid_resolution = float(max(grid_resolution, 1.0e-6))
        self.agent_radius = float(max(agent_radius, 1.0e-6))
        self.role_cooldown_steps = int(max(role_cooldown_steps, 1))
        self.role_gap_threshold = float(max(role_gap_threshold, 0.0))
        self.lidar_range = float(max(lidar_range, 1.0e-6))
        self.lidar_rays = int(max(lidar_rays, 1))
        self.trace_dir = trace_dir
        self.deterministic = bool(deterministic)
        self.fallback_enabled = bool(fallback_enabled)
        self.fallback_only = bool(fallback_only)
        self.obs_dim = 14 + self.lidar_rays
        self.action_dim = 2
        self.robot_ids = list(DEFAULT_ROBOT_IDS)
        self._device_name = "cpu"
        self._actor = None
        self._obs_norm_mean: np.ndarray | None = None
        self._obs_norm_var: np.ndarray | None = None
        self._obs_norm_count: float | None = None
        self._load_error: str | None = None
        self._obs_map = ObstacleMap(
            map_half=self.map_half,
            resolution=self.grid_resolution,
            agent_radius=self.agent_radius,
        )
        self._roles: np.ndarray | None = None
        self._steps_since_switch = 0
        self._close_frames = 0
        self._capture_radius = DEFAULT_CAPTURE_RADIUS
        self._face_capture_thresh = DEFAULT_FACE_CAPTURE_THRESH
        self._agent_max_omega = DEFAULT_AGENT_MAX_OMEGA
        self._committed_sg: list[np.ndarray | None] = [None] * len(self.robot_ids)
        self._sg_age: list[int] = [0] * len(self.robot_ids)
        self._sg_committed_target: list[np.ndarray | None] = [None] * len(self.robot_ids)
        self._was_locked: list[bool] = [False] * len(self.robot_ids)
        self._trace_lock = threading.Lock()
        self._trace_seq = 0
        self._spawn_snapshot_written = False
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._load_policy()

    @property
    def mode(self) -> str:
        if self.fallback_only:
            return "fallback_only"
        return "checkpoint" if self._actor is not None else "fallback"

    @property
    def status(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "checkpoint": str(self.checkpoint),
            "checkpoint_loaded": self._actor is not None,
            "load_error": self._load_error,
            "obs_dim": self.obs_dim,
            "role_cooldown_steps": self.role_cooldown_steps,
            "role_gap_threshold": self.role_gap_threshold,
            "lidar_rays": self.lidar_rays,
            "lidar_range": self.lidar_range,
            "trace_dir": str(self.trace_dir) if self.trace_dir is not None else None,
            "action_semantics": "world_frame_relative_offset_xy",
            "capture_radius": self._capture_radius,
            "face_capture_thresh": self._face_capture_thresh,
            "agent_max_omega": self._agent_max_omega,
            "fallback_only": self.fallback_only,
        }

    def _write_trace(self, record: dict[str, Any]) -> None:
        if self.trace_dir is None:
            return
        trace_path = self.trace_dir / "marl_inference.jsonl"
        with self._trace_lock:
            self._trace_seq += 1
            record["trace_seq"] = self._trace_seq
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _maybe_write_spawn_snapshot(self, robots: dict[str, Any], intruder: dict[str, Any]) -> None:
        if self._spawn_snapshot_written:
            return
        self._write_trace(
            {
                "event_type": "spawn_snapshot",
                "wall_time": time.time(),
                "checkpoint": str(self.checkpoint),
                "policy_mode": self.mode,
                "robots": robots,
                "intruder": intruder,
            }
        )
        self._spawn_snapshot_written = True

    def _load_policy(self) -> None:
        if self.fallback_only:
            self._actor = None
            self._load_error = None
            return

        checkpoint_path = self.checkpoint
        if not checkpoint_path.exists() and checkpoint_path == DEFAULT_CHECKPOINT and LEGACY_CHECKPOINT.exists():
            checkpoint_path = LEGACY_CHECKPOINT
            self.checkpoint = checkpoint_path

        if not checkpoint_path.exists():
            self._load_error = f"checkpoint not found: {checkpoint_path}"
            return

        try:
            import torch

            from marl.policies.actor import Actor

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            actor = Actor(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=64,
                map_half=self.action_limit,
            )
            actor.load_state_dict(checkpoint["actor"])
            actor.eval()
            self._actor = actor
            mean = checkpoint.get("obs_norm_mean")
            var = checkpoint.get("obs_norm_var")
            count = checkpoint.get("obs_norm_count")
            if mean is not None and var is not None and count is not None:
                self._obs_norm_mean = np.asarray(mean, dtype=np.float32).reshape(self.obs_dim)
                self._obs_norm_var = np.asarray(var, dtype=np.float32).reshape(self.obs_dim)
                self._obs_norm_count = float(count)
            self._load_error = None
        except Exception as exc:  # pragma: no cover - narrow runtime integration path
            self._actor = None
            self._load_error = str(exc)

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self._obs_norm_mean is None or self._obs_norm_var is None or self._obs_norm_count is None:
            return obs.astype(np.float32, copy=False)
        std = np.sqrt(np.maximum(self._obs_norm_var, 1.0e-8))
        return ((obs - self._obs_norm_mean) / std).astype(np.float32, copy=False)

    def _ensure_roles(self, agent_positions: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(agent_positions - target_pos[None, :], axis=1)
        if self._roles is None or self._roles.shape != (len(self.robot_ids),):
            if dists[0] <= dists[1]:
                self._roles = np.array([PURSUER, ENCIRCLER], dtype=np.int64)
            else:
                self._roles = np.array([ENCIRCLER, PURSUER], dtype=np.int64)
            self._steps_since_switch = 0
            return self._roles.copy()

        self._steps_since_switch += 1
        gap = abs(float(dists[0]) - float(dists[1]))
        if self._steps_since_switch >= self.role_cooldown_steps and gap >= self.role_gap_threshold:
            desired = np.array(
                [PURSUER, ENCIRCLER] if dists[0] <= dists[1] else [ENCIRCLER, PURSUER],
                dtype=np.int64,
            )
            if not np.array_equal(desired, self._roles):
                self._roles = desired
                self._steps_since_switch = 0
        return self._roles.copy()

    def _compute_lidar(self, pos_xy: np.ndarray) -> np.ndarray:
        px = float(pos_xy[0])
        py = float(pos_xy[1])
        lidar_angles = [i * (2.0 * math.pi / self.lidar_rays) for i in range(self.lidar_rays)]
        return np.asarray(
            [
                self._obs_map.ray_cast(px, py, angle, max_range=self.lidar_range)
                for angle in lidar_angles
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _angle_wrap(value: float) -> float:
        return (float(value) + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _robot_yaw(robot: dict[str, Any]) -> float:
        yaw = robot.get("yaw", 0.0)
        return float(yaw) if isinstance(yaw, int | float) else 0.0

    def _find_reachable_near(self, ideal_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        if not self._obs_map.is_collision(float(ideal_pos[0]), float(ideal_pos[1])):
            return ideal_pos
        base_v = ideal_pos - target_pos
        base_dist = float(np.linalg.norm(base_v))
        if base_dist < 0.1:
            return ideal_pos
        base_angle = float(np.arctan2(base_v[1], base_v[0]))
        for d_factor in (1.0, 0.85, 0.7, 0.55):
            for d_deg in (5, -5, 10, -10, 20, -20, 30, -30, 45, -45, 60, -60, 80, -80):
                ang = base_angle + np.radians(d_deg)
                r = base_dist * d_factor
                cand = target_pos + r * np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
                if not self._obs_map.is_collision(float(cand[0]), float(cand[1])):
                    return cand.astype(np.float32, copy=False)
        return ideal_pos

    def _compute_closing_actions(
        self,
        agent_pos: np.ndarray,
        target_pos: np.ndarray,
        capture_radius: float,
        agent_yaw: np.ndarray | None = None,
        face_thresh: float = 0.5,
    ) -> tuple[np.ndarray, list[bool]]:
        target_dist = capture_radius * 0.7
        align_radius = 0.45
        cap_safe = capture_radius - 0.1
        overshoot = target_dist - 0.3
        face_lock_th = max(0.7, face_thresh)

        d = np.linalg.norm(agent_pos - target_pos[None, :], axis=1)
        pursuer_idx = int(np.argmin(d))
        encircler_idx = 1 - pursuer_idx
        actions = np.zeros((2, 2), dtype=np.float64)
        lock_flags = [False, False]

        v_p = agent_pos[pursuer_idx] - target_pos
        n_p = float(np.linalg.norm(v_p))
        v_e = agent_pos[encircler_idx] - target_pos
        n_e = float(np.linalg.norm(v_e))
        if n_p > 0.05 and n_e > 0.05:
            cos_pe = float(np.dot(v_p, v_e) / (n_p * n_e))
        else:
            cos_pe = 1.0
        encircled = cos_pe <= 0.0

        if n_p > 0.1:
            ideal_p = target_pos + (v_p / n_p) * target_dist
        else:
            ideal_p = agent_pos[pursuer_idx].copy()
        ideal_p = self._find_reachable_near(ideal_p.astype(np.float32, copy=False), target_pos)

        encircler_on_detour = False
        if n_p > 0.1:
            u_p = v_p / n_p
            if cos_pe <= -0.5:
                ideal_e = target_pos - u_p * target_dist
            elif cos_pe <= 0.3:
                ideal_e = target_pos - u_p * 2.0
                encircler_on_detour = True
            else:
                if n_e > 0.05:
                    cross = float(v_p[0] * v_e[1] - v_p[1] * v_e[0])
                    sign = 1.0 if abs(cross) < 0.05 or cross >= 0.0 else -1.0
                else:
                    sign = 1.0
                perp = np.array([-u_p[1], u_p[0]], dtype=np.float32) * sign
                ideal_e = target_pos + perp * (capture_radius + 0.3)
                encircler_on_detour = True
        else:
            ideal_e = target_pos + np.array([target_dist, 0.0], dtype=np.float32)
        ideal_e = self._find_reachable_near(ideal_e.astype(np.float32, copy=False), target_pos)

        ideals = {pursuer_idx: ideal_p, encircler_idx: ideal_e}
        lockable = {pursuer_idx: True, encircler_idx: not encircler_on_detour}

        cf = [1.0, 1.0]
        if agent_yaw is not None:
            for i in range(2):
                to_t = target_pos - agent_pos[i]
                nt = float(np.linalg.norm(to_t))
                if nt > 1.0e-3:
                    head = np.array([np.cos(float(agent_yaw[i])), np.sin(float(agent_yaw[i]))], dtype=np.float32)
                    cf[i] = float(np.dot(head, to_t / nt))

        for i in range(2):
            gap_to_slot = float(np.linalg.norm(agent_pos[i] - ideals[i]))
            d_to_target = float(d[i])
            slot_close = gap_to_slot <= align_radius
            in_safe_zone = d_to_target <= cap_safe
            too_close = d_to_target <= overshoot
            lock_now = (
                (slot_close and in_safe_zone and lockable[i])
                or (in_safe_zone and encircled)
                or (too_close and encircled)
            )
            if lock_now:
                lock_flags[i] = True
                actions[i] = np.zeros(2, dtype=np.float64)
            else:
                actions[i] = (ideals[i] - agent_pos[i]).astype(np.float64, copy=False)

        for i in range(2):
            mag = float(np.linalg.norm(actions[i]))
            if mag > SG_POLICY_CAP:
                actions[i] = actions[i] / mag * SG_POLICY_CAP
        return actions.astype(np.float32, copy=False), lock_flags

    def _fallback_should_activate(self, agent_positions: np.ndarray, target_pos: np.ndarray) -> bool:
        d_to_target = np.linalg.norm(agent_positions - target_pos[None, :], axis=1)
        max_d = float(d_to_target.max())
        if max_d <= 6.0:
            self._close_frames += 1
        else:
            self._close_frames = 0
        cf = self._close_frames
        return bool(
            (max_d <= 2.0 and cf >= 1)
            or (max_d <= 3.0 and cf >= 5)
            or (max_d <= 4.0 and cf >= 10)
            or (max_d <= 6.0 and cf >= 20)
        )

    def _build_final_subgoals(
        self,
        robots: dict[str, Any],
        target_pos: np.ndarray,
        raw_actions: np.ndarray,
        fallback_active: bool,
        lock_flags: list[bool],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        subgoals: dict[str, Any] = {}
        debug: dict[str, Any] = {
            "fallback_active": bool(fallback_active),
            "close_frames": int(self._close_frames),
            "lock_flags": list(lock_flags),
            "sg_age": list(self._sg_age),
            "sg_commit_frames": SG_COMMIT_FRAMES,
        }
        for i, robot_id in enumerate(self.robot_ids):
            robot_pos = np.asarray(_as_xy(robots[robot_id]["position"], f"robots.{robot_id}.position"), dtype=np.float32)
            if lock_flags[i]:
                self._committed_sg[i] = None
                self._sg_age[i] = 0
                self._sg_committed_target[i] = None
                self._was_locked[i] = True
                offset = np.zeros(2, dtype=np.float32)
                subgoal_world = robot_pos.copy()
                look_at = [round(float(target_pos[0]), 4), round(float(target_pos[1]), 4)]
            else:
                sg_cap = SG_FALLBACK_CAP if fallback_active else SG_POLICY_CAP
                raw = np.array(raw_actions[i], dtype=np.float64)
                mag = float(np.linalg.norm(raw))
                if mag > sg_cap:
                    raw = (raw / mag) * sg_cap
                candidate_sg = _clip_world(robot_pos + raw.astype(np.float32, copy=False), self.map_half)
                committed = self._committed_sg[i]
                should_recommit = False
                if committed is None:
                    should_recommit = True
                elif self._was_locked[i]:
                    should_recommit = True
                elif self._sg_age[i] >= SG_COMMIT_FRAMES:
                    should_recommit = True
                elif float(np.linalg.norm(robot_pos - committed)) <= SG_REACHED_THRESH:
                    should_recommit = True
                else:
                    committed_target = self._sg_committed_target[i]
                    if committed_target is None or float(np.linalg.norm(target_pos - committed_target)) > SG_TARGET_DRIFT:
                        should_recommit = True
                if should_recommit:
                    self._committed_sg[i] = candidate_sg.copy()
                    self._sg_age[i] = 0
                    self._sg_committed_target[i] = target_pos.copy()
                    self._was_locked[i] = False
                else:
                    self._sg_age[i] += 1
                subgoal_world = self._committed_sg[i].copy() if self._committed_sg[i] is not None else candidate_sg.copy()
                offset = (subgoal_world - robot_pos).astype(np.float32, copy=False)
                look_at = None
            subgoals[robot_id] = {
                "subgoal": [round(float(subgoal_world[0]), 4), round(float(subgoal_world[1]), 4)],
                "offset": [round(float(offset[0]), 4), round(float(offset[1]), 4)],
                "lock_only": bool(lock_flags[i]),
                "look_at": look_at,
            }
        debug["sg_age"] = list(self._sg_age)
        return subgoals, debug

    def _build_observation(self, robots: dict[str, Any], intruder: dict[str, Any]) -> np.ndarray:
        target_pos = np.asarray(_as_xy(intruder["position"], "intruder.position"), dtype=np.float32)
        target_vel = np.asarray(_as_xy(intruder["velocity"], "intruder.velocity"), dtype=np.float32)
        agent_positions = np.stack(
            [
                np.asarray(_as_xy(robots[robot_id]["position"], f"robots.{robot_id}.position"), dtype=np.float32)
                for robot_id in self.robot_ids
            ],
            axis=0,
        )
        roles = self._ensure_roles(agent_positions, target_pos)

        obs = np.zeros((2, self.obs_dim), dtype=np.float32)
        for i, robot_id in enumerate(self.robot_ids):
            teammate_id = self.robot_ids[1 - i]
            robot = robots[robot_id]
            teammate = robots[teammate_id]
            self_pos = np.asarray(_as_xy(robot["position"], f"robots.{robot_id}.position"), dtype=np.float32)
            self_vel = np.asarray(_as_xy(robot["velocity"], f"robots.{robot_id}.velocity"), dtype=np.float32)
            mate_pos = np.asarray(_as_xy(teammate["position"], f"robots.{teammate_id}.position"), dtype=np.float32)
            mate_vel = np.asarray(_as_xy(teammate["velocity"], f"robots.{teammate_id}.velocity"), dtype=np.float32)
            lidar = self._compute_lidar(self_pos)
            obs[i] = np.concatenate(
                [
                    np.array([float(i)], dtype=np.float32),
                    np.array([float(roles[i])], dtype=np.float32),
                    self_pos / self.map_half,
                    self_vel / self.vel_scale,
                    mate_pos / self.map_half,
                    mate_vel / self.vel_scale,
                    target_pos / self.map_half,
                    target_vel / self.vel_scale,
                    lidar,
                ]
            )
        return obs

    def _fallback_offsets(self, robots: dict[str, Any], intruder: dict[str, Any]) -> np.ndarray:
        target_pos = np.asarray(_as_xy(intruder["position"], "intruder.position"), dtype=np.float32)
        offsets: list[np.ndarray] = []
        for robot_id in self.robot_ids:
            robot_pos = np.asarray(_as_xy(robots[robot_id]["position"], f"robots.{robot_id}.position"), dtype=np.float32)
            offset = _clip_offset_norm(target_pos - robot_pos, self.action_limit)
            offsets.append(offset.astype(np.float32, copy=False))
        return np.stack(offsets, axis=0)

    def infer(self, robots: dict[str, Any], intruder: dict[str, Any]) -> dict[str, Any]:
        for robot_id in self.robot_ids:
            if robot_id not in robots:
                raise ValueError(f"missing robot state for {robot_id}")

        self._maybe_write_spawn_snapshot(robots, intruder)
        obs = self._build_observation(robots, intruder)
        roles = self._roles.copy() if isinstance(self._roles, np.ndarray) else np.zeros(len(self.robot_ids), dtype=np.int64)
        obs_norm = self._normalize_obs(obs)
        target_pos = np.asarray(_as_xy(intruder["position"], "intruder.position"), dtype=np.float32)
        agent_positions = np.stack(
            [
                np.asarray(_as_xy(robots[robot_id]["position"], f"robots.{robot_id}.position"), dtype=np.float32)
                for robot_id in self.robot_ids
            ],
            axis=0,
        )
        agent_yaws = np.asarray([self._robot_yaw(robots[robot_id]) for robot_id in self.robot_ids], dtype=np.float32)
        if self.fallback_only:
            policy_action_offsets = self._fallback_offsets(robots, intruder)
        elif self._actor is None:
            if not self.fallback_enabled:
                raise RuntimeError(self._load_error or "MARL checkpoint is unavailable")
            policy_action_offsets = self._fallback_offsets(robots, intruder)
        else:
            import torch

            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs_norm, dtype=torch.float32)
                actions, _ = self._actor.get_action(obs_tensor, deterministic=self.deterministic)
            policy_action_offsets = actions.cpu().numpy().astype(np.float32, copy=False)

        chosen_action_offsets = np.asarray(policy_action_offsets, dtype=np.float32, copy=True)
        fallback_active = (
            not self.fallback_only
            and self._actor is not None
            and self.fallback_enabled
            and self._fallback_should_activate(agent_positions, target_pos)
        )
        lock_flags = [False, False]
        if fallback_active:
            chosen_action_offsets, lock_flags = self._compute_closing_actions(
                agent_positions,
                target_pos,
                self._capture_radius,
                agent_yaw=agent_yaws,
                face_thresh=self._face_capture_thresh,
            )

        subgoals, debug = self._build_final_subgoals(
            robots,
            target_pos,
            chosen_action_offsets,
            fallback_active=fallback_active,
            lock_flags=lock_flags,
        )

        for i, robot_id in enumerate(self.robot_ids):
            subgoals[robot_id].update(
                {
                    "role": int(roles[i]),
                    "role_name": "pursuer" if int(roles[i]) == PURSUER else "encircler",
                    "mode": "closing_fallback" if fallback_active else "intercept",
                    "priority": 1,
                }
            )

        result = {
            "roles": {
                robot_id: {
                    "role": int(roles[i]),
                    "role_name": "pursuer" if int(roles[i]) == PURSUER else "encircler",
                }
                for i, robot_id in enumerate(self.robot_ids)
            },
            "subgoals": subgoals,
            "policy_mode": self.mode,
            "decision_mode": "closing_fallback" if fallback_active else self.mode,
            **debug,
            "action_semantics": "world_frame_relative_offset_xy",
        }
        self._write_trace(
            {
                "event_type": "inference",
                "wall_time": time.time(),
                "checkpoint": str(self.checkpoint),
                "policy_mode": self.mode,
                "robots": robots,
                "intruder": intruder,
                "roles": result["roles"],
                "obs": np.round(obs, 6).tolist(),
                "obs_norm": np.round(obs_norm, 6).tolist(),
                "policy_action_offsets": np.round(policy_action_offsets, 6).tolist(),
                "chosen_action_offsets": np.round(chosen_action_offsets, 6).tolist(),
                "result": result,
            }
        )
        role_summary = ", ".join(f"{rid}:{info['role_name']}" for rid, info in result["roles"].items())
        print(
            "[marl-trace] "
            f"mode={result['decision_mode']} "
            f"close_frames={self._close_frames} "
            f"roles={{{role_summary}}}"
        )
        return result


class MarlRequestHandler(BaseHTTPRequestHandler):
    runner: MarlPolicyRunner | None = None

    def do_GET(self) -> None:
        if self.path == "/health":
            assert self.runner is not None
            self._send_json({"status": "ok", "owner": "marl", **self.runner.status})
            return
        self.send_error(404, "unknown endpoint")

    def do_POST(self) -> None:
        if self.path != "/act":
            self.send_error(404, "unknown endpoint")
            return

        try:
            payload = self._read_json()
            robots = payload["robots"]
            intruder = payload["intruder"]
            assert self.runner is not None
            result = self.runner.infer(robots, intruder)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, AssertionError, RuntimeError) as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        response = {
            "source_module": "marl",
            "timestamp": float(payload.get("timestamp", 0.0)),
            **result,
        }
        self._send_json(response)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        return json.loads(raw_body.decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[marl] {self.address_string()} - {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MARL decision module adapter service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8892)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--map-half", type=float, default=10.0)
    parser.add_argument("--vel-scale", type=float, default=0.75)
    parser.add_argument("--action-limit", type=float, default=3.0)
    parser.add_argument("--grid-resolution", type=float, default=DEFAULT_GRID_RESOLUTION)
    parser.add_argument("--agent-radius", type=float, default=DEFAULT_AGENT_RADIUS)
    parser.add_argument("--role-cooldown-steps", type=int, default=DEFAULT_ROLE_COOLDOWN_STEPS)
    parser.add_argument("--role-gap-threshold", type=float, default=DEFAULT_ROLE_GAP_THRESHOLD)
    parser.add_argument("--lidar-range", type=float, default=DEFAULT_LIDAR_RANGE)
    parser.add_argument("--lidar-rays", type=int, default=DEFAULT_LIDAR_RAYS)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--no-fallback", action="store_true", help="fail if checkpoint cannot be loaded")
    parser.add_argument("--fallback-only", action="store_true", help="skip policy inference and use fallback logic only")
    args = parser.parse_args()

    MarlRequestHandler.runner = MarlPolicyRunner(
        checkpoint=args.checkpoint,
        map_half=args.map_half,
        vel_scale=args.vel_scale,
        action_limit=args.action_limit,
        grid_resolution=args.grid_resolution,
        agent_radius=args.agent_radius,
        role_cooldown_steps=args.role_cooldown_steps,
        role_gap_threshold=args.role_gap_threshold,
        lidar_range=args.lidar_range,
        lidar_rays=args.lidar_rays,
        trace_dir=args.trace_dir,
        deterministic=args.deterministic,
        fallback_enabled=not args.no_fallback,
        fallback_only=args.fallback_only,
    )
    server = ThreadingHTTPServer((args.host, args.port), MarlRequestHandler)
    status = MarlRequestHandler.runner.status
    print(
        f"[marl] listening on http://{args.host}:{args.port} "
        f"mode={status['mode']} checkpoint={args.checkpoint}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
