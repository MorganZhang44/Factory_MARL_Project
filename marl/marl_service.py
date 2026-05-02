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

If a MAPPO checkpoint is available, the service loads the shared actor and runs
deterministic inference. If not, it falls back to a simple intercept offset so
the module boundary can still be exercised end to end.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from typing import Any

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_CHECKPOINT = PROJECT_ROOT / "results" / "checkpoints" / "final.pt"
LEGACY_CHECKPOINT = MODULE_ROOT / "checkpoints" / "mappo_latest.pt"
DEFAULT_ROBOT_IDS = ("agent_1", "agent_2")


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
        deterministic: bool = True,
        fallback_enabled: bool = True,
    ) -> None:
        self.checkpoint = checkpoint
        self.map_half = float(map_half)
        self.vel_scale = float(max(vel_scale, 1.0e-6))
        self.action_limit = float(max(action_limit, 1.0e-6))
        self.deterministic = bool(deterministic)
        self.fallback_enabled = bool(fallback_enabled)
        self.obs_dim = 13
        self.action_dim = 2
        self.robot_ids = list(DEFAULT_ROBOT_IDS)
        self._device_name = "cpu"
        self._actor = None
        self._obs_norm_mean: np.ndarray | None = None
        self._obs_norm_var: np.ndarray | None = None
        self._obs_norm_count: float | None = None
        self._load_error: str | None = None
        self._load_policy()

    @property
    def mode(self) -> str:
        return "checkpoint" if self._actor is not None else "fallback"

    @property
    def status(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "checkpoint": str(self.checkpoint),
            "checkpoint_loaded": self._actor is not None,
            "load_error": self._load_error,
            "obs_dim": self.obs_dim,
            "action_semantics": "world_frame_relative_offset_xy",
        }

    def _load_policy(self) -> None:
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

    def _build_observation(self, robots: dict[str, Any], intruder: dict[str, Any]) -> np.ndarray:
        target_pos = np.asarray(_as_xy(intruder["position"], "intruder.position"), dtype=np.float32)
        target_vel = np.asarray(_as_xy(intruder["velocity"], "intruder.velocity"), dtype=np.float32)

        obs = np.zeros((2, self.obs_dim), dtype=np.float32)
        for i, robot_id in enumerate(self.robot_ids):
            teammate_id = self.robot_ids[1 - i]
            robot = robots[robot_id]
            teammate = robots[teammate_id]
            self_pos = np.asarray(_as_xy(robot["position"], f"robots.{robot_id}.position"), dtype=np.float32)
            self_vel = np.asarray(_as_xy(robot["velocity"], f"robots.{robot_id}.velocity"), dtype=np.float32)
            mate_pos = np.asarray(_as_xy(teammate["position"], f"robots.{teammate_id}.position"), dtype=np.float32)
            mate_vel = np.asarray(_as_xy(teammate["velocity"], f"robots.{teammate_id}.velocity"), dtype=np.float32)
            obs[i] = np.concatenate(
                [
                    np.array([float(i)], dtype=np.float32),
                    self_pos / self.map_half,
                    self_vel / self.vel_scale,
                    mate_pos / self.map_half,
                    mate_vel / self.vel_scale,
                    target_pos / self.map_half,
                    target_vel / self.vel_scale,
                ]
            )
        return obs

    def _fallback_offsets(self, robots: dict[str, Any], intruder: dict[str, Any]) -> np.ndarray:
        target_pos = np.asarray(_as_xy(intruder["position"], "intruder.position"), dtype=np.float32)
        offsets: list[np.ndarray] = []
        for robot_id in self.robot_ids:
            robot_pos = np.asarray(_as_xy(robots[robot_id]["position"], f"robots.{robot_id}.position"), dtype=np.float32)
            offset = target_pos - robot_pos
            offset = np.clip(offset, -self.action_limit, self.action_limit)
            offsets.append(offset.astype(np.float32, copy=False))
        return np.stack(offsets, axis=0)

    def infer(self, robots: dict[str, Any], intruder: dict[str, Any]) -> dict[str, Any]:
        for robot_id in self.robot_ids:
            if robot_id not in robots:
                raise ValueError(f"missing robot state for {robot_id}")

        obs = self._build_observation(robots, intruder)
        if self._actor is None:
            if not self.fallback_enabled:
                raise RuntimeError(self._load_error or "MARL checkpoint is unavailable")
            action_offsets = self._fallback_offsets(robots, intruder)
        else:
            import torch

            obs_norm = self._normalize_obs(obs)
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs_norm, dtype=torch.float32)
                actions, _ = self._actor.get_action(obs_tensor, deterministic=self.deterministic)
            action_offsets = actions.cpu().numpy().astype(np.float32, copy=False)

        subgoals: dict[str, Any] = {}
        for i, robot_id in enumerate(self.robot_ids):
            robot_pos = np.asarray(_as_xy(robots[robot_id]["position"], f"robots.{robot_id}.position"), dtype=np.float32)
            offset = np.clip(action_offsets[i], -self.action_limit, self.action_limit)
            subgoal_world = _clip_world(robot_pos + offset, self.map_half)
            subgoals[robot_id] = {
                "subgoal": [round(float(subgoal_world[0]), 4), round(float(subgoal_world[1]), 4)],
                "offset": [round(float(offset[0]), 4), round(float(offset[1]), 4)],
                "mode": "intercept",
                "priority": 1,
            }

        return {
            "subgoals": subgoals,
            "policy_mode": self.mode,
            "action_semantics": "world_frame_relative_offset_xy",
        }


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
    parser.add_argument("--vel-scale", type=float, default=1.5)
    parser.add_argument("--action-limit", type=float, default=5.0)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--no-fallback", action="store_true", help="fail if checkpoint cannot be loaded")
    args = parser.parse_args()

    MarlRequestHandler.runner = MarlPolicyRunner(
        checkpoint=args.checkpoint,
        map_half=args.map_half,
        vel_scale=args.vel_scale,
        action_limit=args.action_limit,
        deterministic=args.deterministic,
        fallback_enabled=not args.no_fallback,
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
