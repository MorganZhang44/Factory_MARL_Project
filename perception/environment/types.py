"""Typed data contracts produced by the simulation environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DogMotionHint:
    """Commanded dog body motion published by the environment layer."""

    linear_velocity: Any
    angular_velocity: Any


@dataclass
class ImuSensorOutput:
    """Raw IMU fields required by dog self-localization."""

    ang_vel_b: Any
    lin_acc_b: Any
    projected_gravity_b: Any
    timestamp: float
    odom_vel_w: Any | None = None
    odom_ang_vel_w: Any | None = None


@dataclass
class LidarSensorOutput:
    """Raw LiDAR/raycaster output for one dog-mounted LiDAR."""

    hit_points: Any
    pos_w: Any
    quat_w: Any


@dataclass
class CameraSensorOutput:
    """Raw camera output for one CCTV or dog-mounted camera."""

    semantic_segmentation: Any
    depth: Any | None
    rgb: Any | None
    info: dict
    intrinsic_matrix: Any | None
    pos_w: Any
    quat_w: Any


@dataclass
class GroundTruthState:
    """Optional simulation-only diagnostic state for one actor."""

    pos: Any
    quat: Any | None = None
    lin_vel: Any | None = None
    ang_vel: Any | None = None


@dataclass
class EnvironmentSensorFrame:
    """A single timestamped sensor packet emitted by the environment layer.

    Perception consumes this object and should not need to directly access the
    Isaac scene. Ground truth is included only for evaluation/logging; it must
    not be used as a measurement input to localization algorithms.
    """

    step: int
    timestamp: float
    dog_imus: dict[str, ImuSensorOutput] = field(default_factory=dict)
    dog_lidars: dict[str, LidarSensorOutput] = field(default_factory=dict)
    cameras: dict[str, CameraSensorOutput] = field(default_factory=dict)
    dog_motion_hints: dict[str, DogMotionHint] = field(default_factory=dict)
    ground_truth: dict[str, GroundTruthState] = field(default_factory=dict)

    def camera_detector_inputs(self) -> tuple[dict[str, dict], dict[str, tuple[Any, Any]]]:
        """Return camera data in the shape expected by CameraDetector."""
        camera_data: dict[str, dict] = {}
        camera_poses: dict[str, tuple[Any, Any]] = {}
        for name, output in self.cameras.items():
            camera_data[name] = {
                "semantic_segmentation": output.semantic_segmentation,
                "depth": output.depth,
                "rgb": output.rgb,
                "info": output.info,
                "intrinsic_matrix": output.intrinsic_matrix,
            }
            camera_poses[name] = (output.pos_w, output.quat_w)
        return camera_data, camera_poses

    def lidar_detector_inputs(self) -> dict[str, dict]:
        """Return LiDAR data in the shape expected by LidarDetector."""
        return {
            name: {
                "hit_points": output.hit_points,
                "pos_w": output.pos_w,
                "quat_w": output.quat_w,
            }
            for name, output in self.dog_lidars.items()
        }

    def dog_imu_inputs(self) -> dict[str, dict]:
        """Return IMU data in the shape expected by DogLocalizer."""
        data: dict[str, dict] = {}
        for name, output in self.dog_imus.items():
            imu = {
                "ang_vel_b": output.ang_vel_b,
                "lin_acc_b": output.lin_acc_b,
                "projected_gravity_b": output.projected_gravity_b,
                "timestamp": output.timestamp,
            }
            if output.odom_vel_w is not None:
                imu["odom_vel_w"] = output.odom_vel_w
            if output.odom_ang_vel_w is not None:
                imu["odom_ang_vel_w"] = output.odom_ang_vel_w
            data[name] = imu
        return data
