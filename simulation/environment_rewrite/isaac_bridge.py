"""Isaac Lab adapter that emits environment-layer sensor frames."""

from __future__ import annotations

from typing import Any

import torch
import numpy as np
from isaaclab.scene import InteractiveScene

from .types import (
    CameraSensorOutput,
    DogMotionHint,
    EnvironmentSensorFrame,
    GroundTruthState,
    ImuSensorOutput,
    LidarSensorOutput,
)


DOG_NAMES = ("go2_dog_1", "go2_dog_2")
DOG_LIDAR_NAMES = ("dog1_lidar", "dog2_lidar")


def get_entity(scene: InteractiveScene, name: str):
    """Return a scene entity from articulations, sensors, or rigid objects."""
    if name in scene.articulations.keys():
        return scene.articulations[name]
    if name in scene.sensors.keys():
        return scene.sensors[name]
    if hasattr(scene, "rigid_objects") and name in scene.rigid_objects.keys():
        return scene.rigid_objects[name]
    return None


def get_scene_keys(scene: InteractiveScene) -> list[str]:
    """Return all known scene entity keys."""
    keys = list(scene.articulations.keys())
    keys += list(scene.sensors.keys())
    if hasattr(scene, "rigid_objects"):
        keys += list(scene.rigid_objects.keys())
    return keys


class IsaacEnvironmentBridge:
    """Collect raw Isaac sensor outputs into environment-frame dataclasses."""

    def __init__(self, scene: InteractiveScene):
        self.scene = scene

    def collect_frame(
        self,
        step: int,
        timestamp: float,
        *,
        motion_hints: dict[str, dict] | None = None,
        include_lidar: bool = True,
        include_cameras: bool = True,
        include_ground_truth: bool = True,
    ) -> EnvironmentSensorFrame:
        """Collect one raw environment sensor frame."""
        dog_motion_hints = self._collect_motion_hints(motion_hints)
        return EnvironmentSensorFrame(
            step=step,
            timestamp=timestamp,
            dog_imus=self.collect_dog_imus(timestamp, dog_motion_hints),
            dog_lidars=self.collect_lidars() if include_lidar else {},
            cameras=self.collect_cameras() if include_cameras else {},
            dog_motion_hints=dog_motion_hints,
            ground_truth=self.collect_ground_truth() if include_ground_truth else {},
        )

    def collect_dog_imus(
        self,
        timestamp: float,
        motion_hints: dict[str, DogMotionHint] | None = None,
    ) -> dict[str, ImuSensorOutput]:
        """Collect IMU outputs from both dogs."""
        from isaaclab.utils.math import quat_apply_inverse

        imu_data: dict[str, ImuSensorOutput] = {}
        for dog_name in DOG_NAMES:
            try:
                imu_name = f"{dog_name}_imu"
                if imu_name in get_scene_keys(self.scene):
                    imu = get_entity(self.scene, imu_name)
                    output = ImuSensorOutput(
                        ang_vel_b=imu.data.ang_vel_b[0],
                        lin_acc_b=imu.data.lin_acc_b[0],
                        projected_gravity_b=imu.data.projected_gravity_b[0],
                        timestamp=timestamp,
                    )
                else:
                    dog = get_entity(self.scene, dog_name)
                    if dog is None:
                        continue
                    base_quat_w = dog.data.root_quat_w[0]
                    body_lin_acc_w = dog.data.body_lin_acc_w[0, 0]
                    body_ang_vel_w = dog.data.body_ang_vel_w[0, 0]
                    gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=base_quat_w.device)
                    output = ImuSensorOutput(
                        ang_vel_b=quat_apply_inverse(base_quat_w.unsqueeze(0), body_ang_vel_w.unsqueeze(0)).squeeze(0),
                        lin_acc_b=quat_apply_inverse(base_quat_w.unsqueeze(0), body_lin_acc_w.unsqueeze(0)).squeeze(0),
                        projected_gravity_b=quat_apply_inverse(base_quat_w.unsqueeze(0), gravity_w.unsqueeze(0)).squeeze(0),
                        timestamp=timestamp,
                    )

                if motion_hints and dog_name in motion_hints:
                    hint = motion_hints[dog_name]
                    device = output.ang_vel_b.device
                    output.odom_vel_w = torch.tensor(
                        hint.linear_velocity,
                        dtype=torch.float32,
                        device=device,
                    )
                    output.odom_ang_vel_w = torch.tensor(
                        hint.angular_velocity,
                        dtype=torch.float32,
                        device=device,
                    )

                imu_data[dog_name] = output
            except (KeyError, AttributeError):
                continue
        return imu_data

    def collect_lidars(self) -> dict[str, LidarSensorOutput]:
        """Collect raw dog-mounted LiDAR ray hits."""
        lidar_data: dict[str, LidarSensorOutput] = {}
        for lidar_name in DOG_LIDAR_NAMES:
            try:
                lidar = self.scene[lidar_name]
                data = lidar.data
                if data.ray_hits_w is None:
                    continue
                lidar_data[lidar_name] = LidarSensorOutput(
                    hit_points=data.ray_hits_w,
                    pos_w=data.pos_w,
                    quat_w=data.quat_w,
                )
            except (KeyError, AttributeError, RuntimeError):
                continue
        return lidar_data

    def collect_cameras(self) -> dict[str, CameraSensorOutput]:
        """Collect raw CCTV and dog camera outputs."""
        camera_data: dict[str, CameraSensorOutput] = {}
        for cam_name in [key for key in get_scene_keys(self.scene) if "cam" in key.lower()]:
            try:
                cam = self.scene[cam_name]
                data = cam.data
                if data.output is None:
                    continue

                semantic = data.output.get("semantic_segmentation")
                depth = data.output.get("depth")
                rgb = data.output.get("rgb")
                if semantic is None:
                    continue

                if "cam_" in cam_name:
                    depth = None

                semantic = self._squeeze_semantic(semantic)
                depth = self._squeeze_depth(depth)
                rgb = self._squeeze_rgb(rgb)

                info: dict[str, Any] = {}
                if data.info is not None and len(data.info) > 0:
                    env_info = data.info[0] if isinstance(data.info, list) else data.info
                    if isinstance(env_info, dict):
                        info = env_info.get("semantic_segmentation", {})

                pos = data.pos_w[0] if data.pos_w is not None else torch.zeros(3, device=semantic.device)
                quat = (
                    data.quat_w_ros[0]
                    if data.quat_w_ros is not None
                    else torch.tensor([1.0, 0.0, 0.0, 0.0], device=semantic.device)
                )
                intrinsic = data.intrinsic_matrices[0] if data.intrinsic_matrices is not None else None

                camera_data[cam_name] = CameraSensorOutput(
                    semantic_segmentation=semantic,
                    depth=depth,
                    rgb=rgb,
                    info=info,
                    intrinsic_matrix=intrinsic,
                    pos_w=pos,
                    quat_w=quat,
                )
            except (KeyError, AttributeError, RuntimeError):
                continue
        return camera_data

    def collect_camera_rgb(self) -> dict[str, Any]:
        """Collect camera RGB frames as uint8 numpy arrays for visualization."""
        images: dict[str, Any] = {}
        for cam_name in [key for key in get_scene_keys(self.scene) if "cam" in key.lower()]:
            try:
                cam = get_entity(self.scene, cam_name)
                rgb = cam.data.output.get("rgb")
                rgb = self._squeeze_rgb(rgb)
                if rgb is None:
                    continue
                rgb_np = rgb[:, :, :3].cpu().numpy()
                if rgb_np.dtype != np.uint8:
                    rgb_np = (rgb_np * 255.0).astype("uint8") if rgb_np.max() <= 1.0 else rgb_np.astype("uint8")
                images[cam_name] = rgb_np
            except (KeyError, AttributeError):
                continue
        return images

    def collect_ground_truth(self) -> dict[str, GroundTruthState]:
        """Collect simulation ground truth for evaluation and API diagnostics."""
        gt: dict[str, GroundTruthState] = {}
        for name in (*DOG_NAMES, "suspect"):
            entity = get_entity(self.scene, name)
            if entity is None:
                continue
            try:
                gt[name] = GroundTruthState(
                    pos=entity.data.root_pos_w[0],
                    quat=entity.data.root_quat_w[0],
                    lin_vel=entity.data.root_lin_vel_w[0],
                    ang_vel=entity.data.root_ang_vel_w[0],
                )
            except AttributeError:
                continue
        return gt

    @staticmethod
    def _collect_motion_hints(motion_hints: dict[str, dict] | None) -> dict[str, DogMotionHint]:
        if not motion_hints:
            return {}
        return {
            dog_name: DogMotionHint(
                linear_velocity=tuple(values.get("linear_velocity", (0.0, 0.0, 0.0))),
                angular_velocity=tuple(values.get("angular_velocity", (0.0, 0.0, 0.0))),
            )
            for dog_name, values in motion_hints.items()
        }

    @staticmethod
    def _squeeze_semantic(semantic):
        if semantic is None:
            return None
        if semantic.dim() == 4:
            return semantic[0, :, :, 0]
        if semantic.dim() == 3:
            return semantic[:, :, 0] if semantic.shape[-1] == 1 else semantic[0]
        return semantic

    @staticmethod
    def _squeeze_depth(depth):
        if depth is None:
            return None
        if depth.dim() == 4:
            return depth[0, :, :, 0]
        if depth.dim() == 3:
            return depth[:, :, 0] if depth.shape[-1] == 1 else depth[0]
        return depth

    @staticmethod
    def _squeeze_rgb(rgb):
        if rgb is None:
            return None
        if rgb.dim() == 4:
            return rgb[0, :, :, :3]
        if rgb.dim() == 3:
            return rgb[:, :, :3]
        return rgb
