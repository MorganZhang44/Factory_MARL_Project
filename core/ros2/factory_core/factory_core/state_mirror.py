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
from sensor_msgs.msg import Image, Imu, LaserScan, PointCloud2


@dataclass
class TimedValue:
    value: Any = None
    stamp_sec: float | None = None
    hz: float | None = None


@dataclass
class RobotMirror:
    pose: TimedValue = field(default_factory=TimedValue)
    camera: TimedValue = field(default_factory=TimedValue)
    depth: TimedValue = field(default_factory=TimedValue)
    semantic: TimedValue = field(default_factory=TimedValue)
    imu: TimedValue = field(default_factory=TimedValue)
    lidar: TimedValue = field(default_factory=TimedValue)
    lidar_points: TimedValue = field(default_factory=TimedValue)
    observation: TimedValue = field(default_factory=TimedValue)
    planning: TimedValue = field(default_factory=TimedValue)
    locomotion: TimedValue = field(default_factory=TimedValue)


@dataclass
class DashboardMirror:
    robots: dict[str, RobotMirror] = field(default_factory=dict)
    cctv_cameras: dict[str, TimedValue] = field(default_factory=dict)
    cctv_semantics: dict[str, TimedValue] = field(default_factory=dict)
    intruders: dict[str, TimedValue] = field(default_factory=dict)
    perception: TimedValue = field(default_factory=TimedValue)
    marl: TimedValue = field(default_factory=TimedValue)
    aggregate_state: TimedValue = field(default_factory=TimedValue)


class StateMirror:
    """In-memory Core-owned mirror for observability clients."""

    def __init__(
        self,
        robot_ids: list[str],
        intruder_ids: list[str],
        stale_after: float,
        topic_prefix: str,
        cctv_ids: list[str] | None = None,
    ) -> None:
        self.robot_ids = robot_ids
        self.intruder_ids = intruder_ids
        self.cctv_ids = list(cctv_ids or [])
        self.stale_after = stale_after
        self.topic_prefix = topic_prefix
        self._state = DashboardMirror(
            robots={robot_id: RobotMirror() for robot_id in robot_ids},
            cctv_cameras={camera_id: TimedValue() for camera_id in self.cctv_ids},
            cctv_semantics={camera_id: TimedValue() for camera_id in self.cctv_ids},
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

    def update_depth(self, robot_id: str, msg: Image, now_sec: float) -> None:
        with self._lock:
            robot = self._state.robots[robot_id]
            robot.depth = _next_timed_value(robot.depth, _depth_payload(msg), now_sec)

    def update_semantic(self, robot_id: str, msg: Image, now_sec: float) -> None:
        with self._lock:
            robot = self._state.robots[robot_id]
            robot.semantic = _next_timed_value(robot.semantic, _semantic_payload(msg), now_sec)

    def update_imu(self, robot_id: str, msg: Imu, now_sec: float) -> None:
        with self._lock:
            robot = self._state.robots[robot_id]
            robot.imu = _next_timed_value(robot.imu, _imu_payload(msg), now_sec)

    def update_cctv_camera(self, camera_id: str, msg: Image, now_sec: float) -> None:
        payload = _camera_payload(msg)
        if payload is None:
            return
        with self._lock:
            if camera_id not in self._state.cctv_cameras:
                self._state.cctv_cameras[camera_id] = TimedValue()
            self._state.cctv_cameras[camera_id] = _next_timed_value(
                self._state.cctv_cameras[camera_id],
                payload,
                now_sec,
            )

    def update_cctv_semantic(self, camera_id: str, msg: Image, now_sec: float) -> None:
        with self._lock:
            if camera_id not in self._state.cctv_semantics:
                self._state.cctv_semantics[camera_id] = TimedValue()
            self._state.cctv_semantics[camera_id] = _next_timed_value(
                self._state.cctv_semantics[camera_id],
                _semantic_payload(msg),
                now_sec,
            )

    def update_lidar(self, robot_id: str, msg: LaserScan, now_sec: float) -> None:
        with self._lock:
            robot = self._state.robots[robot_id]
            robot.lidar = _next_timed_value(robot.lidar, _lidar_payload(msg), now_sec)

    def update_lidar_points(self, robot_id: str, msg: PointCloud2, now_sec: float) -> None:
        with self._lock:
            robot = self._state.robots[robot_id]
            robot.lidar_points = _next_timed_value(robot.lidar_points, _point_cloud_payload(msg), now_sec)

    def update_locomotion_observation(self, robot_id: str, payload: dict[str, Any], now_sec: float) -> None:
        parsed = _locomotion_observation_payload(payload)
        with self._lock:
            robot = self._state.robots[robot_id]
            robot.observation = _next_timed_value(robot.observation, parsed, now_sec)

    def update_planning_output(
        self,
        robot_id: str,
        planning: dict[str, Any],
        now_sec: float,
    ) -> None:
        with self._lock:
            robot = self._state.robots[robot_id]
            robot.planning = _next_timed_value(robot.planning, planning, now_sec)

    def update_locomotion_output(
        self,
        robot_id: str,
        locomotion: dict[str, Any],
        now_sec: float,
    ) -> None:
        with self._lock:
            robot = self._state.robots[robot_id]
            robot.locomotion = _next_timed_value(robot.locomotion, locomotion, now_sec)

    def update_perception(self, payload: dict[str, Any], now_sec: float) -> None:
        with self._lock:
            self._state.perception = _next_timed_value(self._state.perception, payload, now_sec)

    def update_marl(self, payload: dict[str, Any], now_sec: float) -> None:
        with self._lock:
            self._state.marl = _next_timed_value(self._state.marl, payload, now_sec)

    def robot_observation_snapshot(self, robot_id: str) -> dict[str, Any]:
        now_sec = time.monotonic()
        with self._lock:
            robot = self._state.robots.get(robot_id)
            if robot is None:
                return {}
            return self._timed_snapshot(robot.observation, now_sec)

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
            cctv_cameras = {
                camera_id: self._timed_snapshot(value, now_sec)
                for camera_id, value in self._state.cctv_cameras.items()
            }
            cctv_semantics = {
                camera_id: self._timed_snapshot(value, now_sec)
                for camera_id, value in self._state.cctv_semantics.items()
            }
            if not include_images:
                for camera in cctv_cameras.values():
                    camera.pop("image", None)
            aggregate_state = self._timed_snapshot(self._state.aggregate_state, now_sec)
        return {
            "server_time": now_sec,
            "topic_prefix": self.topic_prefix,
            "aggregate_state_seen": aggregate_state["seen"],
            "aggregate_state": aggregate_state,
            "perception": self._timed_snapshot(self._state.perception, now_sec),
            "marl": self._timed_snapshot(self._state.marl, now_sec),
            "robots": robots,
            "cctv_cameras": cctv_cameras,
            "cctv_semantics": cctv_semantics,
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

        lidar_points = self._timed_snapshot(robot.lidar_points, now_sec)
        if isinstance(lidar_points.get("points_xyz"), list):
            points_xyz = lidar_points["points_xyz"]
            if max_lidar_points > 0 and len(points_xyz) > max_lidar_points:
                step = max(1, len(points_xyz) // max_lidar_points)
                lidar_points["points_xyz"] = points_xyz[::step][:max_lidar_points]

        return {
            "pose": self._timed_snapshot(robot.pose, now_sec),
            "camera": camera,
            "depth": self._timed_snapshot(robot.depth, now_sec),
            "semantic": self._timed_snapshot(robot.semantic, now_sec),
            "imu": self._timed_snapshot(robot.imu, now_sec),
            "lidar": lidar,
            "lidar_points": lidar_points,
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


def _depth_payload(msg: Image) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "frame_id": msg.header.frame_id,
        "width": int(msg.width),
        "height": int(msg.height),
        "encoding": msg.encoding,
        "is_bigendian": bool(msg.is_bigendian),
        "step": int(msg.step),
    }
    if msg.encoding != "32FC1":
        return payload

    expected = int(msg.width) * int(msg.height)
    data = np.frombuffer(msg.data, dtype=np.float32)
    if data.size < expected:
        payload["valid_count"] = 0
        return payload

    values = data[:expected]
    finite = values[np.isfinite(values)]
    payload["valid_count"] = int(finite.size)
    if finite.size > 0:
        payload["min_m"] = round(float(np.min(finite)), 3)
        payload["max_m"] = round(float(np.max(finite)), 3)
        payload["mean_m"] = round(float(np.mean(finite)), 3)
    return payload


def _semantic_payload(msg: Image) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "frame_id": msg.header.frame_id,
        "width": int(msg.width),
        "height": int(msg.height),
        "encoding": msg.encoding,
        "is_bigendian": bool(msg.is_bigendian),
        "step": int(msg.step),
    }
    if msg.encoding != "32SC1":
        return payload

    expected = int(msg.width) * int(msg.height)
    data = np.frombuffer(msg.data, dtype=np.int32)
    if data.size < expected:
        payload["unique_labels"] = []
        return payload

    labels, counts = np.unique(data[:expected], return_counts=True)
    order = np.argsort(counts)[::-1][:16]
    payload["unique_labels"] = [
        {"label": int(labels[index]), "pixels": int(counts[index])}
        for index in order
    ]
    return payload


def _imu_payload(msg: Imu) -> dict[str, Any]:
    return {
        "frame_id": msg.header.frame_id,
        "orientation": [
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
        ],
        "angular_velocity": [
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ],
        "linear_acceleration": [
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ],
    }


def _point_cloud_payload(msg: PointCloud2) -> dict[str, Any]:
    points = _point_cloud_xyz(msg)
    fields = [
        {
            "name": field.name,
            "offset": int(field.offset),
            "datatype": int(field.datatype),
            "count": int(field.count),
        }
        for field in msg.fields
    ]
    return {
        "frame_id": msg.header.frame_id,
        "width": int(msg.width),
        "height": int(msg.height),
        "point_step": int(msg.point_step),
        "row_step": int(msg.row_step),
        "is_dense": bool(msg.is_dense),
        "fields": fields,
        "point_count": int(points.shape[0]),
        "points_xyz": points.astype(float).round(3).tolist(),
    }


def _point_cloud_xyz(msg: PointCloud2) -> np.ndarray:
    if msg.width == 0 or msg.height == 0 or msg.point_step <= 0 or not msg.data:
        return np.empty((0, 3), dtype=np.float32)

    offsets = {
        axis: _point_field_offset(msg, axis)
        for axis in ("x", "y", "z")
    }
    if any(offset is None for offset in offsets.values()):
        return np.empty((0, 3), dtype=np.float32)

    point_count = int(msg.width) * int(msg.height)
    max_offset = max(int(offset) for offset in offsets.values()) + 4
    if int(msg.point_step) < max_offset:
        return np.empty((0, 3), dtype=np.float32)

    data = bytes(msg.data)
    if len(data) < point_count * int(msg.point_step):
        return np.empty((0, 3), dtype=np.float32)

    columns = []
    for axis in ("x", "y", "z"):
        column = np.ndarray(
            shape=(point_count,),
            dtype=np.float32,
            buffer=data,
            offset=int(offsets[axis]),
            strides=(int(msg.point_step),),
        )
        columns.append(column.copy())
    points = np.stack(columns, axis=1)
    valid = np.all(np.isfinite(points), axis=1)
    return points[valid]


def _point_field_offset(msg: PointCloud2, name: str) -> int | None:
    for field in msg.fields:
        if field.name == name and int(field.datatype) == 7 and int(field.count) >= 1:
            return int(field.offset)
    return None


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
