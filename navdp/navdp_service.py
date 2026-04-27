#!/usr/bin/env python3
"""NavDP module HTTP adapter.

Version 1 keeps the module boundary alive without depending on Core internals:
Core sends a robot state plus a world-frame subgoal, this service returns a
world-frame waypoint path. A real NavDP model can replace the planner behind
the same endpoint later.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = MODULE_ROOT / "checkpoints" / "navdp-cross-modal.ckpt"
VENDOR_DIR = MODULE_ROOT / "vendor" / "navdp_baseline"


def _as_xy(value: Any, field_name: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) < 2:
        raise ValueError(f"{field_name} must be a 2D coordinate")
    return float(value[0]), float(value[1])


def _straight_line_waypoints(
    start: tuple[float, float],
    goal: tuple[float, float],
    max_step: float,
) -> list[list[float]]:
    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    distance = math.hypot(dx, dy)
    if distance < 1.0e-6:
        return [[round(start[0], 4), round(start[1], 4)]]

    steps = max(1, int(math.ceil(distance / max(max_step, 1.0e-3))))
    return [
        [
            round(start[0] + dx * i / steps, 4),
            round(start[1] + dy * i / steps, 4),
        ]
        for i in range(steps + 1)
    ]


class NavDPRequestHandler(BaseHTTPRequestHandler):
    max_step = 0.5
    planner_mode = "auto"
    checkpoint = DEFAULT_CHECKPOINT
    device = "cuda:0"
    stop_threshold = -0.5
    _real_planner: "RealNavDPPlanner | None" = None

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"status": "ok", "owner": "navdp"})
            return
        self.send_error(404, "unknown endpoint")

    def do_POST(self) -> None:
        if self.path != "/plan":
            self.send_error(404, "unknown endpoint")
            return

        try:
            payload = self._read_json()
            robot_id = str(payload["robot_id"])
            robot_state = payload["robot_state"]
            start = _as_xy(robot_state["position"], "robot_state.position")
            goal = _as_xy(payload["subgoal"], "subgoal")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        planner = "straight_line_v1"
        try:
            if self.planner_mode != "straight":
                planner = self._real_navdp_plan(payload, start, goal)
                waypoints = planner["waypoints"]
                planner = planner["planner"]
            else:
                waypoints = _straight_line_waypoints(start, goal, self.max_step)
        except Exception as exc:
            if self.planner_mode == "real":
                self._send_json({"error": f"real NavDP failed: {exc}"}, status=500)
                return
            print(f"[navdp] real planner unavailable, falling back to straight line: {exc}")
            waypoints = _straight_line_waypoints(start, goal, self.max_step)

        self._send_json(
            {
                "robot_id": robot_id,
                "waypoints": waypoints,
                "source_module": "navdp",
                "planner": planner,
            }
        )

    def _real_navdp_plan(
        self,
        payload: dict[str, Any],
        start: tuple[float, float],
        goal: tuple[float, float],
    ) -> dict[str, Any]:
        sensors = payload.get("sensors", {})
        rgb_payload = sensors.get("rgb")
        depth_payload = sensors.get("depth")
        if not isinstance(rgb_payload, dict) or not isinstance(depth_payload, dict):
            raise ValueError("real NavDP needs sensors.rgb and sensors.depth")

        if self._real_planner is None:
            self.__class__._real_planner = RealNavDPPlanner(
                checkpoint=self.checkpoint,
                device=self.device,
                stop_threshold=self.stop_threshold,
            )

        rgb = _decode_rgb(rgb_payload)
        depth = _decode_depth(depth_payload)
        if rgb.shape[:2] != depth.shape[:2]:
            raise ValueError(f"RGB/depth size mismatch: {rgb.shape[:2]} vs {depth.shape[:2]}")

        local_goal_payload = payload.get("local_goal")
        if isinstance(local_goal_payload, list | tuple) and len(local_goal_payload) >= 2:
            local_goal = np.array(
                [[float(local_goal_payload[0]), float(local_goal_payload[1]), 0.0]],
                dtype=np.float32,
            )
        else:
            local_goal = np.array([[goal[0] - start[0], goal[1] - start[1], 0.0]], dtype=np.float32)

        trajectory = self._real_planner.plan(local_goal, rgb, depth)
        local_waypoints = [
            [round(float(point[0]), 4), round(float(point[1]), 4)]
            for point in trajectory
        ]
        robot_yaw = float(payload.get("robot_yaw", 0.0))
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        waypoints = [
            [
                round(start[0] + cos_yaw * float(point[0]) - sin_yaw * float(point[1]), 4),
                round(start[1] + sin_yaw * float(point[0]) + cos_yaw * float(point[1]), 4),
            ]
            for point in trajectory
        ]
        if not waypoints:
            waypoints = _straight_line_waypoints(start, goal, self.max_step)
        return {
            "waypoints": waypoints,
            "local_waypoints": local_waypoints,
            "planner": "navdp_real_pointgoal",
        }

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
        print(f"[navdp] {self.address_string()} - {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NavDP module adapter service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8889)
    parser.add_argument("--max-step", type=float, default=0.5)
    parser.add_argument("--planner", choices=["auto", "real", "straight"], default="auto")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stop-threshold", type=float, default=-0.5)
    args = parser.parse_args()

    NavDPRequestHandler.max_step = args.max_step
    NavDPRequestHandler.planner_mode = args.planner
    NavDPRequestHandler.checkpoint = args.checkpoint
    NavDPRequestHandler.device = args.device
    NavDPRequestHandler.stop_threshold = args.stop_threshold
    server = ThreadingHTTPServer((args.host, args.port), NavDPRequestHandler)
    print(f"[navdp] listening on http://{args.host}:{args.port} planner={args.planner}")
    server.serve_forever()


def _decode_rgb(payload: dict[str, Any]) -> np.ndarray:
    height = int(payload["height"])
    width = int(payload["width"])
    encoding = str(payload["encoding"]).lower()
    raw = base64.b64decode(payload["data_base64"])
    channels = 3
    array = np.frombuffer(raw, dtype=np.uint8)
    expected = height * width * channels
    if array.size < expected:
        raise ValueError("RGB image payload is shorter than expected")
    image = array[:expected].reshape(height, width, channels)
    if encoding == "bgr8":
        image = image[:, :, ::-1]
    elif encoding != "rgb8":
        raise ValueError(f"Unsupported RGB encoding: {encoding}")
    return image.astype(np.uint8, copy=False)


def _decode_depth(payload: dict[str, Any]) -> np.ndarray:
    height = int(payload["height"])
    width = int(payload["width"])
    encoding = str(payload["encoding"]).lower()
    raw = base64.b64decode(payload["data_base64"])
    if encoding == "32fc1":
        array = np.frombuffer(raw, dtype="<f4")
    elif encoding == "16uc1":
        array = np.frombuffer(raw, dtype="<u2").astype(np.float32) / 1000.0
    else:
        raise ValueError(f"Unsupported depth encoding: {encoding}")
    expected = height * width
    if array.size < expected:
        raise ValueError("Depth image payload is shorter than expected")
    depth = array[:expected].reshape(height, width).astype(np.float32, copy=False)
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


def _default_intrinsic(width: int, height: int) -> np.ndarray:
    fx = width * 24.0 / 20.955
    fy = fx
    cx = width * 0.5
    cy = height * 0.5
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


class RealNavDPPlanner:
    def __init__(self, checkpoint: Path, device: str, stop_threshold: float) -> None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"NavDP checkpoint does not exist: {checkpoint}")
        if str(VENDOR_DIR) not in sys.path:
            sys.path.insert(0, str(VENDOR_DIR))
        from policy_agent import NavDP_Agent

        self._agent_cls = NavDP_Agent
        self._agent = None
        self._checkpoint = checkpoint
        self._device = device
        self._stop_threshold = stop_threshold
        self._intrinsic_shape: tuple[int, int] | None = None

    def plan(self, point_goal: np.ndarray, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        import cv2
        import torch

        device = self._device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        height, width = rgb.shape[:2]
        if self._agent is None or self._intrinsic_shape != (width, height):
            intrinsic = _default_intrinsic(width, height)
            self._agent = self._agent_cls(
                intrinsic,
                image_size=224,
                memory_size=8,
                predict_size=24,
                temporal_depth=16,
                heads=8,
                token_dim=384,
                navi_model=str(self._checkpoint),
                device=device,
            )
            self._agent.reset(batch_size=1, threshold=self._stop_threshold)
            self._intrinsic_shape = (width, height)

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        trajectory, _, _, _ = self._agent.step_pointgoal(
            point_goal.astype(np.float32),
            bgr[None, ...],
            depth[None, ..., None].astype(np.float32),
        )
        return np.asarray(trajectory[0], dtype=np.float32)


if __name__ == "__main__":
    main()
