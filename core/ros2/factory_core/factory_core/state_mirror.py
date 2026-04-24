from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from geometry_msgs.msg import PoseStamped
from PIL import Image as PillowImage
from sensor_msgs.msg import Image, LaserScan


@dataclass
class TimedValue:
    value: Any = None
    stamp_sec: float | None = None
    hz: float | None = None


@dataclass
class RobotMirror:
    pose: TimedValue = field(default_factory=TimedValue)
    camera: TimedValue = field(default_factory=TimedValue)
    lidar: TimedValue = field(default_factory=TimedValue)
    observation: TimedValue = field(default_factory=TimedValue)
    planning: TimedValue = field(default_factory=TimedValue)
    locomotion: TimedValue = field(default_factory=TimedValue)


@dataclass
class DashboardMirror:
    robots: dict[str, RobotMirror] = field(default_factory=dict)
    intruders: dict[str, TimedValue] = field(default_factory=dict)
    aggregate_state: TimedValue = field(default_factory=TimedValue)


class StateMirror:
    """In-memory Core-owned mirror for observability clients."""

    def __init__(self, robot_ids: list[str], intruder_ids: list[str], stale_after: float, topic_prefix: str) -> None:
        self.robot_ids = robot_ids
        self.intruder_ids = intruder_ids
        self.stale_after = stale_after
        self.topic_prefix = topic_prefix
        self._state = DashboardMirror(
            robots={robot_id: RobotMirror() for robot_id in robot_ids},
            intruders={intruder_id: TimedValue() for intruder_id in intruder_ids},
        )
        import threading

        self._lock = threading.Lock()

    def update_aggregate(self, payload: str, now_sec: float) -> None:
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            value = {"raw": payload}
        with self._lock:
            self._state.aggregate_state = _next_timed_value(self._state.aggregate_state, value, now_sec)

    def update_robot_pose(self, robot_id: str, msg: PoseStamped, now_sec: float) -> None:
        with self._lock:
            robot = self._state.robots[robot_id]
            robot.pose = _next_timed_value(robot.pose, _pose_payload(msg), now_sec)

    def update_intruder_pose(self, intruder_id: str, msg: PoseStamped, now_sec: float) -> None:
        with self._lock:
            self._state.intruders[intruder_id] = _next_timed_value(
                self._state.intruders[intruder_id],
                _pose_payload(msg),
                now_sec,
            )

    def update_camera(self, robot_id: str, msg: Image, now_sec: float) -> None:
        payload = _camera_payload(msg)
        if payload is None:
            return
        with self._lock:
            robot = self._state.robots[robot_id]
            robot.camera = _next_timed_value(robot.camera, payload, now_sec)

    def update_lidar(self, robot_id: str, msg: LaserScan, now_sec: float) -> None:
        with self._lock:
            robot = self._state.robots[robot_id]
            robot.lidar = _next_timed_value(robot.lidar, _lidar_payload(msg), now_sec)

    def update_locomotion_observation(self, robot_id: str, payload: dict[str, Any], now_sec: float) -> None:
        parsed = _locomotion_observation_payload(payload)
        with self._lock:
            robot = self._state.robots[robot_id]
            robot.observation = _next_timed_value(robot.observation, parsed, now_sec)

    def update_control_outputs(
        self,
        robot_id: str,
        planning: dict[str, Any],
        locomotion: dict[str, Any],
        now_sec: float,
    ) -> None:
        with self._lock:
            robot = self._state.robots[robot_id]
            robot.planning = _next_timed_value(robot.planning, planning, now_sec)
            robot.locomotion = _next_timed_value(robot.locomotion, locomotion, now_sec)

    def snapshot(self, include_images: bool = True, max_lidar_points: int = 220) -> dict[str, Any]:
        now_sec = time.monotonic()
        with self._lock:
            robots = {
                robot_id: self._robot_snapshot(robot, now_sec, include_images, max_lidar_points)
                for robot_id, robot in self._state.robots.items()
            }
            intruders = {
                intruder_id: {"pose": self._timed_snapshot(value, now_sec)}
                for intruder_id, value in self._state.intruders.items()
            }
            aggregate_state = self._timed_snapshot(self._state.aggregate_state, now_sec)
        return {
            "server_time": now_sec,
            "topic_prefix": self.topic_prefix,
            "aggregate_state_seen": aggregate_state["seen"],
            "aggregate_state": aggregate_state,
            "robots": robots,
            "intruders": intruders,
        }

    def _robot_snapshot(
        self,
        robot: RobotMirror,
        now_sec: float,
        include_images: bool,
        max_lidar_points: int,
    ) -> dict[str, Any]:
        camera = self._timed_snapshot(robot.camera, now_sec)
        if not include_images:
            camera.pop("image", None)

        lidar = self._timed_snapshot(robot.lidar, now_sec)
        if isinstance(lidar.get("points"), list):
            points = lidar["points"]
            if max_lidar_points > 0 and len(points) > max_lidar_points:
                step = max(1, len(points) // max_lidar_points)
                lidar["points"] = points[::step][:max_lidar_points]

        return {
            "pose": self._timed_snapshot(robot.pose, now_sec),
            "camera": camera,
            "lidar": lidar,
            "observation": self._timed_snapshot(robot.observation, now_sec),
            "planning": self._timed_snapshot(robot.planning, now_sec),
            "locomotion": self._timed_snapshot(robot.locomotion, now_sec),
        }

    def _timed_snapshot(self, item: TimedValue, now_sec: float) -> dict[str, Any]:
        seen = item.stamp_sec is not None
        age = None if item.stamp_sec is None else now_sec - item.stamp_sec
        data = item.value if item.value is not None else {}
        if isinstance(data, dict):
            payload = dict(data)
        else:
            payload = {"value": data}
        return {
            **payload,
            "seen": seen,
            "fresh": bool(seen and age is not None and age <= self.stale_after),
            "age_sec": age,
            "hz": item.hz,
        }


def _next_timed_value(previous: TimedValue, value: Any, now_sec: float) -> TimedValue:
    hz = previous.hz
    if previous.stamp_sec is not None:
        delta = now_sec - previous.stamp_sec
        if delta > 1.0e-6:
            sample_hz = 1.0 / delta
            hz = sample_hz if hz is None else (0.7 * hz + 0.3 * sample_hz)
    return TimedValue(value=value, stamp_sec=now_sec, hz=hz)


def _pose_payload(msg: PoseStamped) -> dict[str, Any]:
    yaw = _quat_to_yaw(
        msg.pose.orientation.w,
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
    )
    return {
        "frame_id": msg.header.frame_id,
        "position": [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ],
        "orientation": [
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ],
        "yaw": yaw,
    }


def _camera_payload(msg: Image) -> dict[str, Any] | None:
    if msg.encoding not in ("rgb8", "bgr8"):
        return None
    channels = 3
    expected = msg.height * msg.width * channels
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if data.size < expected:
        return None
    image = data[:expected].reshape((msg.height, msg.width, channels))
    if msg.encoding == "bgr8":
        image = image[:, :, ::-1]

    max_width = 320
    pil_image = PillowImage.fromarray(image, mode="RGB")
    if pil_image.width > max_width:
        new_height = max(1, int(pil_image.height * max_width / pil_image.width))
        pil_image = pil_image.resize((max_width, new_height))

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=72)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "frame_id": msg.header.frame_id,
        "width": msg.width,
        "height": msg.height,
        "encoding": msg.encoding,
        "image": f"data:image/jpeg;base64,{encoded}",
    }


def _lidar_payload(msg: LaserScan) -> dict[str, Any]:
    ranges = np.asarray(msg.ranges, dtype=np.float32)
    valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
    angles = msg.angle_min + np.arange(ranges.size, dtype=np.float32) * msg.angle_increment
    points = np.stack([ranges[valid] * np.cos(angles[valid]), ranges[valid] * np.sin(angles[valid])], axis=1)
    return {
        "frame_id": msg.header.frame_id,
        "point_count": int(points.shape[0]),
        "range_min": msg.range_min,
        "range_max": msg.range_max,
        "points": points.astype(float).round(3).tolist(),
    }


def _locomotion_observation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("observation", payload)
    observation = np.asarray(raw, dtype=np.float32).reshape(-1)
    if observation.size < 48:
        padded = np.zeros((48,), dtype=np.float32)
        padded[: observation.size] = observation
        observation = padded
    joint_names = [
        "FL_hip",
        "FR_hip",
        "RL_hip",
        "RR_hip",
        "FL_thigh",
        "FR_thigh",
        "RL_thigh",
        "RR_thigh",
        "FL_calf",
        "FR_calf",
        "RL_calf",
        "RR_calf",
    ]
    joint_pos_rel = observation[12:24].astype(float).round(5).tolist()
    joint_vel_rel = observation[24:36].astype(float).round(5).tolist()
    last_action = observation[36:48].astype(float).round(5).tolist()
    return {
        "schema": payload.get("schema"),
        "timestamp": payload.get("timestamp"),
        "base_linear_velocity": observation[0:3].astype(float).round(5).tolist(),
        "base_angular_velocity": observation[3:6].astype(float).round(5).tolist(),
        "projected_gravity": observation[6:9].astype(float).round(5).tolist(),
        "command_slot": observation[9:12].astype(float).round(5).tolist(),
        "joint_names": joint_names,
        "joint_position_rel": joint_pos_rel,
        "joint_velocity_rel": joint_vel_rel,
        "last_action": last_action,
        "planar_speed": float(np.linalg.norm(observation[0:2])),
    }


def _quat_to_yaw(w: float, x: float, y: float, z: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))
