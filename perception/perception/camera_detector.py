# Copyright (c) 2026, Multi-Agent Surveillance Project
# Camera-based suspect detection using semantic segmentation + depth back-projection.

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .transforms import pixel_to_camera, camera_to_world, quat_apply, transform_points, unproject_depth


# The semantic ID assigned to "suspect" class in Isaac Lab.
# This will be determined at runtime from the segmentation info dict.
SUSPECT_SEMANTIC_ID = None  # Set dynamically


@dataclass
class CameraDetection:
    """Result of a camera-based suspect detection."""
    camera_name: str
    detected: bool
    position_world: Optional[torch.Tensor] = None   # (3,) world coordinates
    confidence: float = 0.0
    num_pixels: int = 0                               # suspect pixel count
    mean_depth: float = 0.0                           # mean depth to suspect
    bbox_2d: Optional[tuple[int, int, int, int]] = None  # (x_min, y_min, x_max, y_max)
    occlusion_score: float = 0.0
    visibility_score: float = 1.0
    used_depth: bool = False
    ray_origin_world: Optional[torch.Tensor] = None
    ray_dir_world: Optional[torch.Tensor] = None


class CameraDetector:
    """Detect and localize suspect from camera images using semantic segmentation + depth.

    Pipeline:
    1. Read semantic segmentation → find pixels labeled as 'suspect'
    2. Read depth buffer → extract depths at suspect pixels
    3. Back-project to 3D camera coordinates using intrinsics
    4. Transform to world coordinates using camera extrinsics (pose)
    5. Return centroid as estimated suspect position
    """

    def __init__(
        self,
        camera_intrinsics: dict[str, tuple[float, float, float, float]] | None = None,
        suspect_class_name: str = "suspect",
        min_pixel_threshold: int = 10,
        max_depth_threshold: float = 30.0,
        nominal_person_height: float = 1.7,
    ):
        """Initialize CameraDetector.

        Camera pose (`pos_w`, `quat_w`) and intrinsics (`intrinsic_matrix`)
        are read from each per-frame `CameraSensorOutput`. The optional
        `camera_intrinsics` dict is a name→(fx, fy, cx, cy) fallback used
        only when a frame omits `intrinsic_matrix`; pass an empty dict (or
        omit) when you want perception to be fully driven by what the
        environment publishes per frame.

        Args:
            camera_intrinsics: Optional fallback intrinsics dict.
            suspect_class_name: Semantic class name for the suspect.
            min_pixel_threshold: Minimum suspect pixels to consider a valid detection.
            max_depth_threshold: Maximum depth (m) to include in detection.
        """
        self.intrinsics = dict(camera_intrinsics) if camera_intrinsics else {}
        self.suspect_class_name = suspect_class_name
        self.min_pixel_threshold = min_pixel_threshold
        self.max_depth_threshold = max_depth_threshold
        self.nominal_person_height = nominal_person_height
        self._suspect_id_cache: dict[str, int] = {}

    def detect(
        self,
        camera_name: str,
        semantic_seg: torch.Tensor,
        depth: torch.Tensor,
        cam_pos: torch.Tensor,
        cam_quat: torch.Tensor,
        info_dict: dict | None = None,
        intrinsic_matrix: torch.Tensor | None = None,
    ) -> CameraDetection:
        """Detect suspect in a single camera's data.

        Uses Isaac Lab's create_pointcloud_from_depth() for back-projection,
        ensuring correct coordinate frame handling.

        Pipeline:
        1. Find suspect pixels via semantic ID
        2. Create masked depth image (only suspect pixels)
        3. Use Isaac Lab API to back-project to 3D world coordinates
        4. Take median of 3D points as position estimate

        Args:
            camera_name: Identifier for this camera.
            semantic_seg: Semantic segmentation image, shape (H, W) int32.
            depth: Depth image, shape (H, W) float32, in meters.
            cam_pos: Camera world position, shape (3,).
            cam_quat: Camera world orientation in OpenGL convention (w, x, y, z), shape (4,).
            info_dict: Optional segmentation info mapping semantic IDs to labels.
            intrinsic_matrix: Camera intrinsic matrix, shape (3, 3). If provided, uses
                Isaac Lab's create_pointcloud_from_depth for precise back-projection.

        Returns:
            CameraDetection with results.
        """
        device = semantic_seg.device

        # Determine suspect semantic ID
        suspect_id = self._resolve_suspect_id(camera_name, semantic_seg, info_dict)
        if suspect_id is None:
            return CameraDetection(
                camera_name=camera_name,
                detected=False,
                confidence=0.0,
            )

        # Find suspect pixels
        suspect_mask = (semantic_seg == suspect_id)
        num_pixels = suspect_mask.sum().item()

        if num_pixels < self.min_pixel_threshold:
            return CameraDetection(
                camera_name=camera_name,
                detected=False,
                num_pixels=num_pixels,
                confidence=0.0,
            )

        if intrinsic_matrix is not None and depth is not None and depth.max() > 0:
            # === Pure-torch back-projection (depth available) ===
            # Create masked depth: inf-out non-suspect pixels
            masked_depth = depth.clone()
            masked_depth[~suspect_mask] = float('inf')
            
            # 1. unproject_depth gives points in the camera optical frame (ROS convention)
            points_cam = unproject_depth(masked_depth, intrinsic_matrix)
            valid_mask = torch.all(torch.isfinite(points_cam), dim=-1)
            points_cam_valid = points_cam[valid_mask]
            
            if points_cam_valid.shape[0] < self.min_pixel_threshold:
                return CameraDetection(camera_name=camera_name, detected=False, num_pixels=num_pixels, confidence=0.0)
            
            # 2. transform_points to map from ROS camera frame to World frame
            points_world = transform_points(points_cam_valid, cam_pos.unsqueeze(0), cam_quat.unsqueeze(0)).squeeze(0)
            
            # --- 3D outlier removal ---
            centroid_initial = points_world.median(dim=0).values
            dists = torch.norm(points_world - centroid_initial.unsqueeze(0), dim=-1)
            median_dist = dists.median()
            inlier_mask = dists < (median_dist * 2.0 + 0.5)
            if inlier_mask.sum() >= self.min_pixel_threshold:
                points_world = points_world[inlier_mask]

            centroid = points_world.median(dim=0).values
            mean_depth = depth[suspect_mask].median().item()
            visibility_score = 1.0
            occlusion_score = 0.0
            used_depth = True

        else:
            # === Monocular Localization: Ground Plane Intersection ===
            # Used when depth is unavailable (e.g. standard CCTV)
            v_coords, u_coords = torch.where(suspect_mask)
            x_min = int(u_coords.min())
            x_max = int(u_coords.max())
            y_min = int(v_coords.min())
            y_max = int(v_coords.max())
            bbox_width = max(1, x_max - x_min + 1)
            bbox_height = max(1, y_max - y_min + 1)
            bbox_area = max(1, bbox_width * bbox_height)
            fill_ratio = float(num_pixels) / float(bbox_area)
            
            # Find the bottom-center of the suspect (the "feet" on the ground)
            u_center = (u_coords.min() + u_coords.max()) / 2.0
            v_bottom = v_coords.max().float()
            v_center = (v_coords.min() + v_coords.max()).float() / 2.0
            h_img, w_img = semantic_seg.shape[-2:]
            lower_band_start = max(y_min, y_max - max(2, int(0.18 * bbox_height)) + 1)
            lower_band = suspect_mask[lower_band_start : y_max + 1, x_min : x_max + 1]
            lower_support = lower_band.float().mean().item() if lower_band.numel() > 0 else 0.0
            height_score = min(1.0, bbox_height / max(1.0, 0.18 * h_img))
            fill_score = min(1.0, fill_ratio / 0.35)
            
            # Get camera intrinsics. Per-frame matrix wins; fall back to the
            # static dict only when the environment didn't publish one. If
            # neither source has this camera we drop the detection rather
            # than throw — the camera roster is dynamic.
            if intrinsic_matrix is not None:
                fx = intrinsic_matrix[0, 0].item()
                fy = intrinsic_matrix[1, 1].item()
                cx = intrinsic_matrix[0, 2].item()
                cy = intrinsic_matrix[1, 2].item()
            elif camera_name in self.intrinsics:
                fx, fy, cx, cy = self.intrinsics[camera_name]
            else:
                return CameraDetection(
                    camera_name=camera_name,
                    detected=False,
                    num_pixels=num_pixels,
                    confidence=0.0,
                    bbox_2d=(x_min, y_min, x_max, y_max),
                )
            
            # 1. Image point to camera ray (Optical Frame: X right, Y down, Z forward)
            rx = (u_center - cx) / fx
            ry = (v_bottom - cy) / fy
            rz = 1.0
            ray_cam = torch.tensor([rx, ry, rz], device=device, dtype=torch.float32)
            
            # 2. Transform ray to World Frame
            # We use the rotation from cam_quat (which maps from ROS camera to World)
            ray_world = quat_apply(cam_quat.unsqueeze(0), ray_cam.unsqueeze(0)).squeeze(0)
            ray_world = ray_world / torch.norm(ray_world)
            
            # 3. Intersect ray with Ground Plane (Z = 0)
            # P = CamPos + t * RayWorld
            # CamPos.z + t * RayWorld.z = 0  =>  t = -CamPos.z / RayWorld.z
            if abs(ray_world[2]) > 1e-6:
                t = -cam_pos[2] / ray_world[2]
                if t > 0:
                    ground_estimate = cam_pos + t * ray_world
                    ground_estimate[2] = 0.9
                else:
                    return CameraDetection(camera_name=camera_name, detected=False, confidence=0.0)
            else:
                return CameraDetection(camera_name=camera_name, detected=False, confidence=0.0)

            center_rx = (u_center - cx) / fx
            center_ry = (v_center - cy) / fy
            center_ray_cam = torch.tensor([center_rx, center_ry, 1.0], device=device, dtype=torch.float32)
            center_ray_world = quat_apply(cam_quat.unsqueeze(0), center_ray_cam.unsqueeze(0)).squeeze(0)
            center_ray_world = center_ray_world / torch.norm(center_ray_world)

            depth_from_height = fy * self.nominal_person_height / max(float(bbox_height), 1.0)
            body_center_cam = torch.tensor(
                [center_rx * depth_from_height, center_ry * depth_from_height, depth_from_height],
                device=device,
                dtype=torch.float32,
            )
            body_center_world = cam_pos + quat_apply(cam_quat.unsqueeze(0), body_center_cam.unsqueeze(0)).squeeze(0)
            body_center_world[2] = 0.9

            foot_depth = float(t.item() if isinstance(t, torch.Tensor) else t)
            if foot_depth > self.max_depth_threshold or depth_from_height > self.max_depth_threshold:
                return CameraDetection(
                    camera_name=camera_name,
                    detected=False,
                    num_pixels=num_pixels,
                    confidence=0.0,
                    mean_depth=max(foot_depth, depth_from_height),
                    bbox_2d=(x_min, y_min, x_max, y_max),
                )

            depth_consistency = min(foot_depth, depth_from_height) / max(foot_depth, depth_from_height, 1.0e-6)
            occlusion_score = float(np.clip(0.65 * (1.0 - depth_consistency) + 0.35 * (1.0 - lower_support), 0.0, 1.0))
            blend_to_height = min(0.85, max(0.0, 0.9 * occlusion_score))
            centroid = (1.0 - blend_to_height) * ground_estimate + blend_to_height * body_center_world

            visibility_score = float(
                np.clip(
                    0.2 + 0.35 * height_score + 0.25 * fill_score + 0.20 * lower_support,
                    0.1,
                    1.0,
                )
            )
            mean_depth = 0.5 * (foot_depth + depth_from_height)
            used_depth = False
            if bbox_height < max(16, int(0.035 * h_img)) or visibility_score < 0.16:
                return CameraDetection(
                    camera_name=camera_name,
                    detected=False,
                    num_pixels=num_pixels,
                    confidence=0.0,
                    mean_depth=mean_depth,
                    bbox_2d=(x_min, y_min, x_max, y_max),
                    occlusion_score=occlusion_score,
                    visibility_score=visibility_score,
                    used_depth=used_depth,
                )
            
        # Get bounding box for UI
        v_coords_all, u_coords_all = torch.where(suspect_mask)
        bbox = (int(u_coords_all.min()), int(v_coords_all.min()), int(u_coords_all.max()), int(v_coords_all.max()))
        base_conf = min(1.0, num_pixels / 300.0)
        confidence = base_conf
        if not used_depth:
            confidence *= visibility_score * max(0.15, 1.0 - 0.7 * occlusion_score)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return CameraDetection(
            camera_name=camera_name,
            detected=True,
            position_world=centroid,
            confidence=confidence,
            num_pixels=num_pixels,
            mean_depth=mean_depth,
            bbox_2d=bbox,
            occlusion_score=occlusion_score,
            visibility_score=visibility_score,
            used_depth=used_depth,
            ray_origin_world=cam_pos.detach().clone() if not used_depth else None,
            ray_dir_world=center_ray_world.detach().clone() if not used_depth else None,
        )

    def detect_all_cameras(
        self,
        camera_data: dict[str, dict],
        camera_poses: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> list[CameraDetection]:
        """Run detection on all cameras.

        Args:
            camera_data: Dict camera_name → {"semantic_segmentation": tensor, "depth": tensor, "info": dict}.
            camera_poses: Dict camera_name → (pos, quat) tensors.

        Returns:
            List of CameraDetection results.
        """
        detections = []
        for cam_name, data in camera_data.items():
            pos, quat = camera_poses[cam_name]
            detection = self.detect(
                camera_name=cam_name,
                semantic_seg=data["semantic_segmentation"],
                depth=data["depth"],
                cam_pos=pos,
                cam_quat=quat,
                info_dict=data.get("info"),
                intrinsic_matrix=data.get("intrinsic_matrix"),
            )
            
            detections.append(detection)
        return detections

    def _resolve_suspect_id(
        self,
        camera_name: str,
        semantic_seg: torch.Tensor,
        info_dict: dict | None,
    ) -> int | None:
        """Resolve the semantic ID for the suspect class."""
        # Check cache first
        if camera_name in self._suspect_id_cache:
            return self._suspect_id_cache[camera_name]

        if info_dict is None:
            return None

        id_to_labels = info_dict.get("idToLabels", {})
        if not id_to_labels:
            return None

        # Strategy 1: exact match on suspect class name
        for sem_id, label_info in id_to_labels.items():
            label_str = ""
            if isinstance(label_info, str):
                label_str = label_info
            elif isinstance(label_info, dict):
                label_str = label_info.get("class", "")
            
            if self.suspect_class_name.lower() in label_str.lower():
                sid = int(sem_id)
                self._suspect_id_cache[camera_name] = sid
                return sid
        
        return None

    def set_suspect_id(self, suspect_id: int, camera_name: str | None = None):
        """Manually set the suspect semantic ID.

        Args:
            suspect_id: The integer semantic ID for the suspect.
            camera_name: If provided, set for specific camera. Otherwise set for all.
        """
        if camera_name:
            self._suspect_id_cache[camera_name] = suspect_id
        else:
            # Will be returned for any camera not in cache
            self._default_suspect_id = suspect_id
