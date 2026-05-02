#!/usr/bin/env python3
"""Perception module HTTP adapter.

Core sends mirrored simulation sensor packets. This service converts those
packets into the existing perception EnvironmentSensorFrame contract and
returns compact pose estimates for Core to mirror and route downstream.
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
import torch


MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

try:
    from environment.static_scene_geometry import _CAMERA_CORNER_SPECS, _move_toward
    from environment.types import (
        CameraSensorOutput,
        DogMotionHint,
        EnvironmentSensorFrame,
        GroundTruthState,
        ImuSensorOutput,
        LidarSensorOutput,
    )
    from perception.pipeline import PerceptionPipeline
except ModuleNotFoundError:
    PROJECT_ROOT = MODULE_ROOT.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from perception.environment.static_scene_geometry import _CAMERA_CORNER_SPECS, _move_toward
    from perception.environment.types import (
        CameraSensorOutput,
        DogMotionHint,
        EnvironmentSensorFrame,
        GroundTruthState,
        ImuSensorOutput,
        LidarSensorOutput,
    )
    from perception.perception.pipeline import PerceptionPipeline


ROBOT_NAME_MAP = {
    "agent_1": "go2_dog_1",
    "agent_2": "go2_dog_2",
}

LIDAR_NAME_MAP = {
    "agent_1": "dog1_lidar",
    "agent_2": "dog2_lidar",
}

DOG_CAMERA_NAME_MAP = {
    "agent_1": "dog1_cam",
    "agent_2": "dog2_cam",
}

DOG_CAMERA_OFFSET = np.array([0.3, 0.0, 0.1], dtype=np.float32)
DOG_LIDAR_OFFSET = np.array([0.0, 0.0, 0.35], dtype=np.float32)
CCTV_HEIGHT = 2.35
CCTV_PITCH_DEG = 25.0


def _quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1.0e-9:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _projected_gravity_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    rot_wb = _quat_wxyz_to_rotmat(quat)
    gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    gravity_b = rot_wb.T @ gravity_w
    norm = np.linalg.norm(gravity_b)
    if norm < 1.0e-9:
        return gravity_w
    return (gravity_b / norm).astype(np.float32)


def _yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def _normalize(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    if norm < 1.0e-8:
        return (1.0, 0.0, 0.0)
    return (vec[0] / norm, vec[1] / norm, vec[2] / norm)


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _quat_from_rotmat(
    rot: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
) -> np.ndarray:
    r00, r01, r02 = rot[0]
    r10, r11, r12 = rot[1]
    r20, r21, r22 = rot[2]
    trace = r00 + r11 + r22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r21 - r12) / s
        y = (r02 - r20) / s
        z = (r10 - r01) / s
    elif r00 > r11 and r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        w = (r21 - r12) / s
        x = 0.25 * s
        y = (r01 + r10) / s
        z = (r02 + r20) / s
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        w = (r02 - r20) / s
        x = (r01 + r10) / s
        y = 0.25 * s
        z = (r12 + r21) / s
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        w = (r10 - r01) / s
        x = (r02 + r20) / s
        y = (r12 + r21) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float32)
    norm = np.linalg.norm(quat)
    return quat if norm < 1.0e-8 else quat / norm


def _camera_intrinsics(width: int, height: int, focal_length_mm: float, aperture_mm: float) -> tuple[float, float, float, float]:
    fx = float(width) * float(focal_length_mm) / float(aperture_mm)
    fy = fx
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    return fx, fy, cx, cy


def _camera_pose_specs() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    specs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for entry in _CAMERA_CORNER_SPECS:
        mount_xy = _move_toward(entry["corner"], entry["look_hint"], entry["mount_inset"])
        position = np.array([mount_xy[0], mount_xy[1], CCTV_HEIGHT], dtype=np.float32)
        dx = entry["look_hint"][0] - mount_xy[0]
        dy = entry["look_hint"][1] - mount_xy[1]
        horizontal_norm = math.hypot(dx, dy)
        pitch_rad = math.radians(CCTV_PITCH_DEG)
        if horizontal_norm < 1.0e-6:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            forward = (
                math.cos(pitch_rad) * dx / horizontal_norm,
                math.cos(pitch_rad) * dy / horizontal_norm,
                -math.sin(pitch_rad),
            )
            world_up = (0.0, 0.0, 1.0)
            y_axis = _normalize(_cross(world_up, forward))
            z_axis = _normalize(_cross(forward, y_axis))
            quat = _quat_from_rotmat(
                (
                    (forward[0], y_axis[0], z_axis[0]),
                    (forward[1], y_axis[1], z_axis[1]),
                    (forward[2], y_axis[2], z_axis[2]),
                )
            )
        specs[entry["name"]] = (position, quat)
    return specs


class PerceptionRuntime:
    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.cctv_specs = _camera_pose_specs()
        self.pipeline = self._build_pipeline()
        self._suspect_id = None
        self._last_frame_timestamp: float | None = None

    def _build_pipeline(self) -> PerceptionPipeline:
        pipeline = PerceptionPipeline(
            camera_intrinsics={
                "dog1_cam": _camera_intrinsics(320, 240, 3.5, 12.0),
                "dog2_cam": _camera_intrinsics(320, 240, 3.5, 12.0),
                **{
                    camera_id: _camera_intrinsics(640, 480, 14.0, 20.955)
                    for camera_id in self.cctv_specs.keys()
                },
            },
            dog_initial_poses={
                "go2_dog_1": (
                    torch.tensor([-2.0, -2.0, 0.42], dtype=torch.float32, device=self.device),
                    torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
                ),
                "go2_dog_2": (
                    torch.tensor([-2.0, 1.6, 0.42], dtype=torch.float32, device=self.device),
                    torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
                ),
            },
            dt=1.0 / 15.0,
        )
        return pipeline

    def estimate(self, payload: dict[str, Any]) -> dict[str, Any]:
        frame = self._build_frame(payload)
        self._sync_pipeline_timestep(frame.timestamp)
        if self._suspect_id is not None:
            self.pipeline.camera_detector.set_suspect_id(self._suspect_id)
        output = self.pipeline.update(frame)
        return self._serialize_output(output, frame)

    def _sync_pipeline_timestep(self, timestamp: float) -> None:
        if self._last_frame_timestamp is None:
            self._last_frame_timestamp = float(timestamp)
            return
        dt = float(timestamp) - float(self._last_frame_timestamp)
        self._last_frame_timestamp = float(timestamp)
        if not math.isfinite(dt) or dt <= 1.0e-5:
            return
        for localizer in self.pipeline.dog_localizers.values():
            localizer.dt = dt

    def _build_frame(self, payload: dict[str, Any]) -> EnvironmentSensorFrame:
        step = int(payload.get("step", 0))
        timestamp = float(payload.get("timestamp", 0.0))
        robots = payload.get("robots", {})
        intruders = payload.get("intruders", {})
        cctv = payload.get("cctv", {})
        simulation_state = payload.get("simulation_state", {})

        dog_imus: dict[str, ImuSensorOutput] = {}
        dog_lidars: dict[str, LidarSensorOutput] = {}
        cameras: dict[str, CameraSensorOutput] = {}
        dog_motion_hints: dict[str, DogMotionHint] = {}
        ground_truth: dict[str, GroundTruthState] = {}

        for robot_id, robot_payload in robots.items():
            dog_name = ROBOT_NAME_MAP.get(robot_id)
            if dog_name is None:
                continue

            pose = robot_payload.get("pose")
            pose_pos, pose_quat = self._decode_pose(pose)
            if pose_pos is None or pose_quat is None:
                continue

            motion_hint_payload = robot_payload.get("motion_hint")
            motion_hint = self._build_motion_hint(motion_hint_payload)
            if motion_hint is not None:
                dog_motion_hints[dog_name] = motion_hint

            lin_vel, ang_vel = self._ground_truth_velocity(simulation_state, "robots", robot_id)
            if lin_vel is None and motion_hint is not None:
                lin_vel = torch.tensor(motion_hint.linear_velocity, dtype=torch.float32, device=self.device)
            if ang_vel is None and motion_hint is not None:
                ang_vel = torch.tensor(motion_hint.angular_velocity, dtype=torch.float32, device=self.device)

            ground_truth[dog_name] = GroundTruthState(
                pos=pose_pos,
                quat=pose_quat,
                lin_vel=lin_vel,
                ang_vel=ang_vel,
            )

            imu_payload = robot_payload.get("sensors", {}).get("imu")
            if isinstance(imu_payload, dict):
                dog_imus[dog_name] = self._build_imu_output(imu_payload, timestamp, motion_hint)

            lidar_payload = robot_payload.get("sensors", {}).get("lidar_points")
            if isinstance(lidar_payload, dict):
                lidar_name = LIDAR_NAME_MAP[robot_id]
                lidar_pos, lidar_quat = self._mounted_sensor_pose(pose_pos, pose_quat, DOG_LIDAR_OFFSET)
                lidar_points = self._decode_point_cloud(lidar_payload)
                if lidar_points.shape[0] > 0:
                    world_hits = self._sensor_points_to_world(lidar_points, lidar_pos, lidar_quat)
                    dog_lidars[lidar_name] = LidarSensorOutput(
                        hit_points=world_hits.unsqueeze(0),
                        pos_w=lidar_pos.unsqueeze(0),
                        quat_w=lidar_quat.unsqueeze(0),
                    )

            sensors = robot_payload.get("sensors", {})
            semantic_payload = sensors.get("semantic_segmentation")
            depth_payload = sensors.get("depth")
            rgb_payload = sensors.get("rgb")
            info_payload = sensors.get("info")
            if isinstance(semantic_payload, dict):
                camera_name = DOG_CAMERA_NAME_MAP[robot_id]
                cam_pos, cam_quat = self._camera_pose_from_payload(
                    rgb_payload,
                    default_pos=pose_pos,
                    default_quat=pose_quat,
                    default_offset=DOG_CAMERA_OFFSET,
                )
                semantic = self._decode_image_tensor(semantic_payload)
                depth = self._decode_image_tensor(depth_payload) if isinstance(depth_payload, dict) else None
                rgb = self._decode_image_tensor(rgb_payload) if isinstance(rgb_payload, dict) else None
                cameras[camera_name] = CameraSensorOutput(
                    semantic_segmentation=semantic,
                    depth=depth,
                    rgb=rgb,
                    info=info_payload if isinstance(info_payload, dict) else {},
                    intrinsic_matrix=self._camera_intrinsic_from_payload(rgb_payload, 320, 240, 3.5, 12.0),
                    pos_w=cam_pos,
                    quat_w=cam_quat,
                )

        for camera_id, camera_payload in cctv.items():
            semantic_payload = camera_payload.get("semantic_segmentation")
            if not isinstance(semantic_payload, dict):
                continue
            semantic = self._decode_image_tensor(semantic_payload)
            rgb_payload = camera_payload.get("rgb")
            rgb = self._decode_image_tensor(rgb_payload) if isinstance(rgb_payload, dict) else None
            info_payload = camera_payload.get("info")
            pos_w, quat_w = self._cctv_pose_from_payload(camera_payload, camera_id)
            cameras[camera_id] = CameraSensorOutput(
                semantic_segmentation=semantic,
                depth=None,
                rgb=rgb,
                info=info_payload if isinstance(info_payload, dict) else {},
                intrinsic_matrix=self._camera_intrinsic_from_payload(rgb_payload, 640, 480, 14.0, 20.955),
                pos_w=pos_w,
                quat_w=quat_w,
            )

        suspect_pose = intruders.get("intruder_1", {}).get("pose")
        suspect_pos, suspect_quat = self._decode_pose(suspect_pose)
        if suspect_pos is not None:
            intruder_lin_vel, intruder_ang_vel = self._ground_truth_velocity(simulation_state, "intruders", "intruder_1")
            ground_truth["suspect"] = GroundTruthState(
                pos=suspect_pos,
                quat=suspect_quat,
                lin_vel=intruder_lin_vel,
                ang_vel=intruder_ang_vel,
            )

        suspect_id = payload.get("semantic_labels", {}).get("suspect_id")
        if isinstance(suspect_id, int):
            self._suspect_id = int(suspect_id)

        return EnvironmentSensorFrame(
            step=step,
            timestamp=timestamp,
            dog_imus=dog_imus,
            dog_lidars=dog_lidars,
            cameras=cameras,
            dog_motion_hints=dog_motion_hints,
            ground_truth=ground_truth,
        )

    def _build_imu_output(
        self,
        imu_payload: dict[str, Any],
        timestamp: float,
        motion_hint: DogMotionHint | None,
    ) -> ImuSensorOutput:
        ang_vel = torch.tensor(imu_payload.get("angular_velocity", [0.0, 0.0, 0.0]), dtype=torch.float32, device=self.device)
        lin_acc = torch.tensor(imu_payload.get("linear_acceleration", [0.0, 0.0, 0.0]), dtype=torch.float32, device=self.device)
        orientation = np.asarray(imu_payload.get("orientation", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        projected_gravity = torch.tensor(
            _projected_gravity_from_quat_wxyz(orientation),
            dtype=torch.float32,
            device=self.device,
        )
        return ImuSensorOutput(
            ang_vel_b=ang_vel,
            lin_acc_b=lin_acc,
            projected_gravity_b=projected_gravity,
            timestamp=timestamp,
            odom_vel_w=(
                torch.tensor(motion_hint.linear_velocity, dtype=torch.float32, device=self.device)
                if motion_hint is not None
                else None
            ),
            odom_ang_vel_w=(
                torch.tensor(motion_hint.angular_velocity, dtype=torch.float32, device=self.device)
                if motion_hint is not None
                else None
            ),
        )

    def _build_motion_hint(self, payload: Any) -> DogMotionHint | None:
        if not isinstance(payload, dict):
            return None
        linear = payload.get("linear_velocity_world")
        angular = payload.get("angular_velocity_world")
        if not isinstance(linear, list | tuple) or len(linear) < 3:
            return None
        if not isinstance(angular, list | tuple) or len(angular) < 3:
            return None
        try:
            linear_velocity = tuple(float(value) for value in linear[:3])
            angular_velocity = tuple(float(value) for value in angular[:3])
        except (TypeError, ValueError):
            return None
        return DogMotionHint(
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
        )

    def _camera_pose_from_payload(
        self,
        payload: Any,
        default_pos: torch.Tensor,
        default_quat: torch.Tensor,
        default_offset: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(payload, dict):
            pos = self._vector3_from_payload(payload.get("pos_w"))
            quat = self._quat_from_payload(payload.get("quat_w"))
            if pos is not None and quat is not None:
                return pos, quat
        return self._mounted_sensor_pose(default_pos, default_quat, default_offset)

    def _cctv_pose_from_payload(self, payload: dict[str, Any], camera_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self._vector3_from_payload(payload.get("pos_w"))
        quat = self._quat_from_payload(payload.get("quat_w"))
        if pos is not None and quat is not None:
            return pos, quat
        pos_w_np, quat_w_np = self.cctv_specs.get(
            camera_id,
            (np.zeros(3, dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        )
        return (
            torch.tensor(pos_w_np, dtype=torch.float32, device=self.device),
            torch.tensor(quat_w_np, dtype=torch.float32, device=self.device),
        )

    def _camera_intrinsic_from_payload(
        self,
        payload: Any,
        width: int,
        height: int,
        focal_length_mm: float,
        aperture_mm: float,
    ) -> torch.Tensor:
        if isinstance(payload, dict):
            matrix = payload.get("intrinsic_matrix") or payload.get("intrinsics")
            if isinstance(matrix, list | tuple):
                try:
                    arr = np.asarray(matrix, dtype=np.float32)
                    if arr.shape == (3, 3):
                        return torch.tensor(arr, dtype=torch.float32, device=self.device)
                except (TypeError, ValueError):
                    pass
        return self._intrinsic_matrix(width, height, focal_length_mm, aperture_mm)

    def _vector3_from_payload(self, value: Any) -> torch.Tensor | None:
        if not isinstance(value, list | tuple) or len(value) < 3:
            return None
        try:
            return torch.tensor([float(value[0]), float(value[1]), float(value[2])], dtype=torch.float32, device=self.device)
        except (TypeError, ValueError):
            return None

    def _quat_from_payload(self, value: Any) -> torch.Tensor | None:
        if not isinstance(value, list | tuple) or len(value) < 4:
            return None
        try:
            quat = np.asarray([float(value[0]), float(value[1]), float(value[2]), float(value[3])], dtype=np.float32)
        except (TypeError, ValueError):
            return None
        norm = float(np.linalg.norm(quat))
        if norm < 1.0e-8:
            return None
        quat /= norm
        return torch.tensor(quat, dtype=torch.float32, device=self.device)

    def _ground_truth_velocity(
        self,
        simulation_state: dict[str, Any],
        entity_group: str,
        entity_id: str,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        entities = simulation_state.get(entity_group)
        if not isinstance(entities, dict):
            return None, None
        entity = entities.get(entity_id)
        if not isinstance(entity, dict):
            return None, None
        velocity = entity.get("velocity")
        angular_velocity = entity.get("angular_velocity")
        lin_vel = None
        ang_vel = None
        if isinstance(velocity, list | tuple) and len(velocity) >= 3:
            lin_vel = torch.tensor(velocity[:3], dtype=torch.float32, device=self.device)
        elif isinstance(velocity, list | tuple) and len(velocity) >= 2:
            lin_vel = torch.tensor([float(velocity[0]), float(velocity[1]), 0.0], dtype=torch.float32, device=self.device)
        if isinstance(angular_velocity, list | tuple) and len(angular_velocity) >= 3:
            ang_vel = torch.tensor(angular_velocity[:3], dtype=torch.float32, device=self.device)
        return lin_vel, ang_vel

    def _decode_pose(self, payload: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not isinstance(payload, dict):
            return None, None
        position = payload.get("position")
        orientation = payload.get("orientation")
        if not isinstance(position, list | tuple) or len(position) < 3:
            return None, None
        if not isinstance(orientation, list | tuple) or len(orientation) < 4:
            orientation = [1.0, 0.0, 0.0, 0.0]
        pos = torch.tensor(position[:3], dtype=torch.float32, device=self.device)
        quat = torch.tensor(orientation[:4], dtype=torch.float32, device=self.device)
        return pos, quat

    def _mounted_sensor_pose(
        self,
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor,
        offset_b: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rot = torch.tensor(_quat_wxyz_to_rotmat(body_quat_w.detach().cpu().numpy()), dtype=torch.float32, device=self.device)
        offset = torch.tensor(offset_b, dtype=torch.float32, device=self.device)
        return body_pos_w + rot @ offset, body_quat_w

    def _sensor_points_to_world(
        self,
        points_sensor: torch.Tensor,
        sensor_pos_w: torch.Tensor,
        sensor_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        rot = torch.tensor(_quat_wxyz_to_rotmat(sensor_quat_w.detach().cpu().numpy()), dtype=torch.float32, device=self.device)
        return points_sensor @ rot.T + sensor_pos_w.unsqueeze(0)

    def _decode_point_cloud(self, payload: dict[str, Any]) -> torch.Tensor:
        width = int(payload.get("width", 0))
        height = int(payload.get("height", 0))
        point_step = int(payload.get("point_step", 0))
        fields = payload.get("fields", [])
        raw = base64.b64decode(payload.get("data_base64", ""))
        point_count = width * height
        if point_count <= 0 or point_step <= 0 or not raw:
            return torch.empty((0, 3), dtype=torch.float32, device=self.device)

        offsets = {}
        for field in fields:
            if field.get("name") in ("x", "y", "z"):
                offsets[field["name"]] = int(field["offset"])
        if len(offsets) != 3:
            return torch.empty((0, 3), dtype=torch.float32, device=self.device)

        expected = point_count * point_step
        if len(raw) < expected:
            return torch.empty((0, 3), dtype=torch.float32, device=self.device)

        array = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(point_count, point_step)
        points = np.stack(
            [
                array[:, offsets["x"] : offsets["x"] + 4].view("<f4").reshape(-1),
                array[:, offsets["y"] : offsets["y"] + 4].view("<f4").reshape(-1),
                array[:, offsets["z"] : offsets["z"] + 4].view("<f4").reshape(-1),
            ],
            axis=1,
        )
        valid = np.isfinite(points).all(axis=1)
        return torch.tensor(points[valid], dtype=torch.float32, device=self.device)

    def _decode_image_tensor(self, payload: dict[str, Any]) -> torch.Tensor | None:
        encoding = str(payload.get("encoding", "")).lower()
        width = int(payload.get("width", 0))
        height = int(payload.get("height", 0))
        raw = base64.b64decode(payload.get("data_base64", ""))
        if width <= 0 or height <= 0 or not raw:
            return None
        if encoding in ("rgb8", "bgr8"):
            array = np.frombuffer(raw, dtype=np.uint8)
            expected = width * height * 3
            if array.size < expected:
                return None
            image = array[:expected].reshape(height, width, 3)
            if encoding == "bgr8":
                image = image[:, :, ::-1].copy()
            return torch.tensor(image, dtype=torch.uint8, device=self.device)
        if encoding == "32fc1":
            array = np.frombuffer(raw, dtype="<f4")
            expected = width * height
            if array.size < expected:
                return None
            return torch.tensor(array[:expected].reshape(height, width), dtype=torch.float32, device=self.device)
        if encoding == "32sc1":
            array = np.frombuffer(raw, dtype="<i4")
            expected = width * height
            if array.size < expected:
                return None
            return torch.tensor(array[:expected].reshape(height, width), dtype=torch.int32, device=self.device)
        return None

    def _intrinsic_matrix(self, width: int, height: int, focal_length_mm: float, aperture_mm: float) -> torch.Tensor:
        fx, fy, cx, cy = _camera_intrinsics(width, height, focal_length_mm, aperture_mm)
        return torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        )

    def _serialize_output(self, output: Any, frame: EnvironmentSensorFrame) -> dict[str, Any]:
        dogs: dict[str, Any] = {}
        for robot_id, dog_name in ROBOT_NAME_MAP.items():
            estimate = output.dogs.get(dog_name)
            if estimate is None:
                dogs[robot_id] = {"detected": False}
                continue
            state = estimate.state or {}
            pos = self._tensor_to_list(state.get("pos"))
            quat = self._tensor_to_list(state.get("quat"))
            vel = self._tensor_to_list(state.get("vel"))
            dogs[robot_id] = {
                "detected": True,
                "localized": bool(state.get("localized", False)),
                "position_world": pos,
                "orientation_world": quat,
                "velocity_world": vel,
                "yaw_rad": self._safe_float(state.get("yaw")),
                "xy_error_m": self._safe_float(estimate.xy_error_m),
                "scan_match_score": self._safe_float(state.get("scan_match_score")),
                "scan_inliers": int(state.get("scan_match_inliers", 0) or 0),
                "used_prediction_only": bool(state.get("used_prediction_only", False)),
            }

        intruder_payload = {"detected": False}
        if output.intruder is not None:
            fusion = output.intruder.fusion_result
            position = None if fusion is None else self._tensor_to_list(getattr(fusion, "position_world", None))
            intruder_payload = {
                "detected": bool(getattr(fusion, "detected", False)) if fusion is not None else False,
                "position_world": position,
                "confidence": self._safe_float(getattr(fusion, "confidence", None)) if fusion is not None else None,
                "camera_detections": len(output.intruder.camera_detections),
                "lidar_detections": len(output.intruder.lidar_detections),
            }

        return {
            "source_module": "perception",
            "pipeline": "perception_pipeline_v1",
            "step": int(output.step),
            "timestamp": float(output.timestamp),
            "dogs": dogs,
            "intruder_estimate": intruder_payload,
            "input_summary": {
                "dog_imus": sorted(frame.dog_imus.keys()),
                "dog_lidars": sorted(frame.dog_lidars.keys()),
                "cameras": sorted(frame.cameras.keys()),
                "ground_truth": sorted(frame.ground_truth.keys()),
            },
        }

    @staticmethod
    def _tensor_to_list(value: Any) -> list[float] | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
        if isinstance(value, np.ndarray):
            return [float(item) for item in value.reshape(-1).tolist()]
        if isinstance(value, list | tuple):
            return [float(item) for item in value]
        return None

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


class PerceptionRequestHandler(BaseHTTPRequestHandler):
    runtime: PerceptionRuntime | None = None

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"status": "ok", "owner": "perception"})
            return
        self.send_error(404, "unknown endpoint")

    def do_POST(self) -> None:
        if self.path != "/estimate":
            self.send_error(404, "unknown endpoint")
            return
        try:
            payload = self._read_json()
            if self.runtime is None:
                raise RuntimeError("perception runtime was not initialized")
            result = self.runtime.estimate(payload)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)
            return
        self._send_json(result)

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
        print(f"[perception] {self.address_string()} - {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Perception module adapter service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    PerceptionRequestHandler.runtime = PerceptionRuntime(device=args.device)
    server = ThreadingHTTPServer((args.host, args.port), PerceptionRequestHandler)
    print(f"[perception] listening on http://{args.host}:{args.port} device={args.device}")
    server.serve_forever()


if __name__ == "__main__":
    main()
