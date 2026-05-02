"""High-level perception package interface.

This module is the boundary between environment sensor packets and estimated
global poses. It consumes EnvironmentSensorFrame objects and produces
PerceptionOutput objects; it does not read from the Isaac scene directly.

All emitted positions, velocities, and orientations are in the global world
frame (matching Isaac Lab's `root_pos_w`, `root_lin_vel_w`, and `root_quat_w`
conventions). Body-frame quantities are not exposed by this layer.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from environment.types import EnvironmentSensorFrame
from environment.static_scene_geometry import get_static_object_positions

from .camera_detector import CameraDetector
from .dog_localizer import DogLocalizer
from .fusion import SensorFusion
from .lidar_detector import LidarDetector
from .scan_matching import build_static_map_from_scene_cfg
from .types import DogPoseEstimate, IntruderPoseEstimate, PerceptionOutput


class PerceptionPipeline:
    """Run dog self-localization and intruder localization from environment frames."""

    def __init__(
        self,
        *,
        dog_initial_poses: dict[str, tuple[Any, Any]],
        dt: float,
        camera_intrinsics: dict[str, tuple[float, float, float, float]] | None = None,
        lidar_mount_height: float = 0.35,
        dog_lidar_names: dict[str, str] | None = None,
    ) -> None:
        # CCTV pose + intrinsic matrix come per-frame from the
        # `CameraSensorOutput` carried by the environment frame, so the
        # pipeline does not need a static camera roster. The optional
        # `camera_intrinsics` dict is forwarded as a fallback used only
        # when a frame omits its `intrinsic_matrix` field.
        self.camera_detector = CameraDetector(
            camera_intrinsics=camera_intrinsics or {},
            suspect_class_name="suspect",
            min_pixel_threshold=10,
            max_depth_threshold=30.0,
        )
        self.lidar_detector = LidarDetector(
            height_min=0.3,
            height_max=2.0,
            cluster_distance=1.5,
            min_cluster_points=3,
            static_object_positions=get_static_object_positions(),
            static_object_radius=1.0,
        )
        self.sensor_fusion = SensorFusion(
            camera_weight=1.1,
            lidar_weight=0.55,
            temporal_alpha=0.7,
            outlier_threshold=3.0,
            history_size=5000,
            camera_gate_distance=1.8,
            lidar_gate_distance=2.8,
        )
        static_map = build_static_map_from_scene_cfg()
        self.dog_localizers = {
            dog_name: DogLocalizer(
                dog_name,
                dt,
                initial_pose=initial_pose,
                lidar_mount_height=lidar_mount_height,
                static_map=static_map,
                max_lidar_speed_mps=1.80,
            )
            for dog_name, initial_pose in dog_initial_poses.items()
        }
        self.dog_lidar_names = dog_lidar_names or {
            "go2_dog_1": "dog1_lidar",
            "go2_dog_2": "dog2_lidar",
        }

    def update(
        self,
        frame: EnvironmentSensorFrame,
        *,
        update_intruder: bool = True,
        update_dogs: bool = True,
    ) -> PerceptionOutput:
        """Consume one environment frame and return all pose estimates."""
        output = PerceptionOutput(step=frame.step, timestamp=frame.timestamp)

        if update_dogs:
            output.dogs = self.update_dogs(frame)

        if update_intruder:
            output.intruder = self.update_intruder(frame)

        return output

    def update_dogs(self, frame: EnvironmentSensorFrame) -> dict[str, DogPoseEstimate]:
        """Update all dog localizers from IMU and optional LiDAR data.

        The `state["pos"]`, `state["vel"]`, and `state["quat"]` returned by
        each localizer are all in the global world frame (same convention as
        Isaac Lab's `root_pos_w` / `root_lin_vel_w` / `root_quat_w`).
        """
        estimates: dict[str, DogPoseEstimate] = {}
        dog_imus = frame.dog_imu_inputs()
        lidar_inputs = frame.lidar_detector_inputs()

        for dog_name, imu_data in dog_imus.items():
            localizer = self.dog_localizers.get(dog_name)
            if localizer is None:
                continue
            lidar_name = self.dog_lidar_names.get(dog_name)
            state = localizer.update(imu_data, lidar_inputs.get(lidar_name) if lidar_name else None)
            gt = frame.ground_truth.get(dog_name)
            xy_error = None
            yaw_err_deg = None
            if gt is not None and "pos" in state:
                est_pos = state["pos"]
                gt_pos = gt.pos.to(est_pos.device) if hasattr(gt.pos, "to") else gt.pos
                xy_error = float(torch.norm(est_pos[:2] - gt_pos[:2]).item())
                if gt.quat is not None and "euler" in state:
                    gt_yaw = _yaw_from_quat_wxyz(gt.quat)
                    est_yaw = float(state["euler"][2])
                    yaw_err_deg = yaw_error_deg(est_yaw, gt_yaw)
            estimates[dog_name] = DogPoseEstimate(
                name=dog_name,
                step=frame.step,
                timestamp=frame.timestamp,
                state=state,
                ground_truth=gt,
                xy_error_m=xy_error,
                yaw_error_deg=yaw_err_deg,
            )
        return estimates

    def update_intruder(self, frame: EnvironmentSensorFrame) -> IntruderPoseEstimate:
        """Estimate intruder global pose from CCTV/dog camera and LiDAR outputs."""
        camera_data, camera_poses = frame.camera_detector_inputs()
        camera_detections = self.camera_detector.detect_all_cameras(camera_data, camera_poses)
        lidar_detections = self.lidar_detector.detect_all_lidars(frame.lidar_detector_inputs())
        gt = frame.ground_truth.get("suspect")
        fusion_result = self.sensor_fusion.fuse(
            camera_detections=camera_detections,
            lidar_detections=lidar_detections,
            ground_truth=gt.pos if gt is not None else None,
            timestamp=frame.timestamp,
            step=frame.step,
        )
        return IntruderPoseEstimate(
            step=frame.step,
            timestamp=frame.timestamp,
            fusion_result=fusion_result,
            camera_detections=camera_detections,
            lidar_detections=lidar_detections,
            ground_truth=gt,
        )


def yaw_error_deg(est_yaw: float, gt_yaw: float) -> float:
    """Smallest absolute yaw error in degrees."""
    delta = math.atan2(math.sin(est_yaw - gt_yaw), math.cos(est_yaw - gt_yaw))
    return abs(math.degrees(delta))


def _yaw_from_quat_wxyz(quat: Any) -> float:
    """Extract world-frame yaw from a (w, x, y, z) quaternion."""
    if hasattr(quat, "detach"):
        q = quat.detach().to(dtype=torch.float32).flatten().tolist()
    else:
        q = list(quat)
    if len(q) < 4:
        return 0.0
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)
