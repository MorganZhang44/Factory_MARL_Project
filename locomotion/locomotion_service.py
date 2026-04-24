#!/usr/bin/env python3
"""Locomotion module HTTP adapter.

Version 1 converts a world-frame waypoint path into a world-frame velocity
command. A learned locomotion policy can replace this controller behind the
same endpoint later.
"""

from __future__ import annotations

import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_POLICY = Path(__file__).resolve().parent / "checkpoints" / "go2_flat_actor_model_499.npz"


def _as_xy(value: Any, field_name: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) < 2:
        raise ValueError(f"{field_name} must be a 2D coordinate")
    return float(value[0]), float(value[1])


def _choose_target(
    position: tuple[float, float],
    waypoints: list[Any],
    lookahead: float,
) -> tuple[float, float]:
    if not waypoints:
        raise ValueError("path.waypoints must not be empty")

    target = _as_xy(waypoints[-1], "path.waypoints[-1]")
    for waypoint in waypoints:
        candidate = _as_xy(waypoint, "path.waypoints[]")
        if math.hypot(candidate[0] - position[0], candidate[1] - position[1]) >= lookahead:
            target = candidate
            break
    return target


def _velocity_to_target(
    position: tuple[float, float],
    target: tuple[float, float],
    max_speed: float,
    stop_distance: float,
) -> list[float]:
    dx = target[0] - position[0]
    dy = target[1] - position[1]
    distance = math.hypot(dx, dy)
    if distance <= stop_distance:
        return [0.0, 0.0]

    speed = min(max_speed, distance)
    return [round(dx / distance * speed, 4), round(dy / distance * speed, 4)]


class LocomotionRequestHandler(BaseHTTPRequestHandler):
    max_speed = 0.8
    lookahead = 0.35
    stop_distance = 0.15
    policy_path = DEFAULT_POLICY
    action_scale = 0.25
    policy: "Go2ActorPolicy | None" = None

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"status": "ok", "owner": "locomotion"})
            return
        self.send_error(404, "unknown endpoint")

    def do_POST(self) -> None:
        if self.path != "/command":
            self.send_error(404, "unknown endpoint")
            return

        try:
            payload = self._read_json()
            robot_id = str(payload["robot_id"])
            position = _as_xy(payload["robot_state"]["position"], "robot_state.position")
            waypoints = payload["path"]["waypoints"]
            target = _choose_target(position, waypoints, self.lookahead)
            velocity = _velocity_to_target(position, target, self.max_speed, self.stop_distance)
            action = self._policy_action(payload, velocity)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        self._send_json(
            {
                "robot_id": robot_id,
                "velocity": velocity,
                "action": action,
                "action_scale": self.action_scale,
                "target": [round(target[0], 4), round(target[1], 4)],
                "source_module": "locomotion",
                "controller": "go2_low_level_policy_v1",
            }
        )

    def _policy_action(self, payload: dict[str, Any], velocity: list[float]) -> list[float] | None:
        observation = payload.get("locomotion_observation")
        if observation is None:
            return None
        obs = np.asarray(observation.get("observation", observation), dtype=np.float32).reshape(-1)
        if obs.size != 48:
            raise ValueError(f"locomotion observation must have 48 values, got {obs.size}")

        command = payload.get("body_velocity_command")
        if command is None:
            command = [velocity[0], velocity[1], 0.0]
        if not isinstance(command, list | tuple) or len(command) < 3:
            raise ValueError("body_velocity_command must have [vx, vy, wz]")
        obs[9:12] = np.asarray(command[:3], dtype=np.float32)

        if self.__class__.policy is None:
            self.__class__.policy = Go2ActorPolicy(self.policy_path)
        return self.__class__.policy(obs).round(5).astype(float).tolist()

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
        print(f"[locomotion] {self.address_string()} - {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Locomotion module adapter service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--max-speed", type=float, default=0.8)
    parser.add_argument("--lookahead", type=float, default=0.35)
    parser.add_argument("--stop-distance", type=float, default=0.15)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--action-scale", type=float, default=0.25)
    args = parser.parse_args()

    LocomotionRequestHandler.max_speed = args.max_speed
    LocomotionRequestHandler.lookahead = args.lookahead
    LocomotionRequestHandler.stop_distance = args.stop_distance
    LocomotionRequestHandler.policy_path = args.policy
    LocomotionRequestHandler.action_scale = args.action_scale
    server = ThreadingHTTPServer((args.host, args.port), LocomotionRequestHandler)
    print(f"[locomotion] listening on http://{args.host}:{args.port} policy={args.policy}")
    server.serve_forever()


class Go2ActorPolicy:
    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Locomotion policy weights not found: {path}")
        weights = np.load(path)
        self.weights = [
            (weights["w0"].astype(np.float32), weights["b0"].astype(np.float32)),
            (weights["w1"].astype(np.float32), weights["b1"].astype(np.float32)),
            (weights["w2"].astype(np.float32), weights["b2"].astype(np.float32)),
            (weights["w3"].astype(np.float32), weights["b3"].astype(np.float32)),
        ]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = obs.astype(np.float32)
        for weight, bias in self.weights[:-1]:
            x = _elu(weight @ x + bias)
        weight, bias = self.weights[-1]
        return weight @ x + bias


def _elu(value: np.ndarray) -> np.ndarray:
    return np.where(value > 0.0, value, np.exp(value) - 1.0).astype(np.float32)


if __name__ == "__main__":
    main()
