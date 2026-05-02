# Copyright (c) 2026, Multi-Agent Surveillance Project
# LiDAR-based suspect detection using point cloud filtering and clustering.

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class LidarDetection:
    """Result of a LiDAR-based suspect detection."""
    lidar_name: str
    detected: bool
    position_world: Optional[torch.Tensor] = None   # (3,) world coordinates
    confidence: float = 0.0
    num_points: int = 0                               # points in suspect cluster
    cluster_std: float = 0.0                          # cluster spread (m)


class LidarDetector:
    """Detect and localize suspect from LiDAR point clouds.

    Pipeline:
    1. Read raycaster hit points (world frame from Isaac Lab)
    2. Filter by height range (suspect should be 0.5m — 2.0m above ground)
    3. Remove known static objects by comparing against a static map
    4. Cluster remaining points (distance-based)
    5. Return centroid of largest non-static cluster as suspect position
    """

    def __init__(
        self,
        height_min: float = 0.3,
        height_max: float = 2.0,
        cluster_distance: float = 1.0,
        min_cluster_points: int = 3,
        static_object_positions: list[tuple[float, float, float]] | None = None,
        static_object_radius: float = 1.5,
    ):
        """Initialize LidarDetector.

        Args:
            height_min: Minimum height above ground (m) for suspect points.
            height_max: Maximum height above ground (m) for suspect points.
            cluster_distance: Maximum distance (m) between points in same cluster.
            min_cluster_points: Minimum points to form a valid cluster.
            static_object_positions: Known positions of static objects to exclude.
            static_object_radius: Radius around static objects to exclude.
        """
        self.height_min = height_min
        self.height_max = height_max
        self.cluster_distance = cluster_distance
        self.min_cluster_points = min_cluster_points
        self.static_positions = static_object_positions or []
        self.static_radius = static_object_radius

    def detect(
        self,
        lidar_name: str,
        hit_points: torch.Tensor,
        sensor_pos: torch.Tensor | None = None,
        sensor_quat: torch.Tensor | None = None,
    ) -> LidarDetection:
        """Detect suspect from a single LiDAR's data.

        Args:
            lidar_name: Identifier for this LiDAR.
            hit_points: Ray hit positions in world frame, shape (B, N, 3) or (N, 3).
                        Invalid hits have inf values.
            sensor_pos: Sensor world position (B, 3) or (3,).
            sensor_quat: Sensor world orientation (w,x,y,z) (B, 4) or (4,).

        Returns:
            LidarDetection with results.
        """
        if hit_points.numel() == 0:
            return LidarDetection(lidar_name=lidar_name, detected=False)

        # Ensure 2D shape: (N, 3)
        if hit_points.dim() == 3:
            hit_points = hit_points.squeeze(0)  # (B, N, 3) -> (N, 3)

        points_world = hit_points

        # Filter by valid hits (invalid rays return inf in Isaac Lab)
        valid_mask = torch.isfinite(points_world).all(dim=-1)
        points_valid = points_world[valid_mask]

        if points_valid.shape[0] == 0:
            return LidarDetection(lidar_name=lidar_name, detected=False)

        # Filter by height (suspect height range)
        z_coords = points_valid[:, 2]
        height_mask = (z_coords >= self.height_min) & (z_coords <= self.height_max)
        points_height = points_valid[height_mask]

        if points_height.shape[0] < self.min_cluster_points:
            return LidarDetection(lidar_name=lidar_name, detected=False)

        # Remove points near known static objects
        points_dynamic = self._remove_static_objects(points_height)

        if points_dynamic.shape[0] < self.min_cluster_points:
            return LidarDetection(lidar_name=lidar_name, detected=False)

        # Cluster the remaining points
        clusters = self._cluster_points(points_dynamic)

        if not clusters:
            return LidarDetection(lidar_name=lidar_name, detected=False)

        # Take the largest cluster as the suspect
        largest_cluster = max(clusters, key=lambda c: c.shape[0])

        if largest_cluster.shape[0] < self.min_cluster_points:
            return LidarDetection(lidar_name=lidar_name, detected=False)

        # Compute centroid
        centroid = largest_cluster.mean(dim=0)
        cluster_std = largest_cluster.std(dim=0).norm().item()

        # Confidence based on cluster quality
        point_conf = min(1.0, largest_cluster.shape[0] / 20.0)
        spread_conf = max(0.0, 1.0 - cluster_std / 2.0)
        confidence = 0.5 * point_conf + 0.5 * spread_conf

        return LidarDetection(
            lidar_name=lidar_name,
            detected=True,
            position_world=centroid,
            confidence=confidence,
            num_points=largest_cluster.shape[0],
            cluster_std=cluster_std,
        )

    def detect_all_lidars(
        self,
        lidar_data: dict[str, dict],
    ) -> list[LidarDetection]:
        """Run detection on all LiDARs.

        Args:
            lidar_data: Dict lidar_name → {"hit_points": tensor, "pos_w": tensor, "quat_w": tensor}.

        Returns:
            List of LidarDetection results.
        """
        detections = []
        for lidar_name, data in lidar_data.items():
            detection = self.detect(
                lidar_name=lidar_name,
                hit_points=data["hit_points"],
                sensor_pos=data.get("pos_w"),
                sensor_quat=data.get("quat_w"),
            )
            detections.append(detection)
        return detections

    def update_static_map(self, positions: list[tuple[float, float, float]]):
        """Update the static object positions.

        Args:
            positions: List of (x, y, z) positions to consider static.
        """
        self.static_positions = positions

    def _remove_static_objects(self, points: torch.Tensor) -> torch.Tensor:
        """Remove points near known static object positions.

        Args:
            points: 3D points, shape (N, 3).

        Returns:
            Filtered points with static objects removed.
        """
        if not self.static_positions:
            return points

        mask = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
        for static_pos in self.static_positions:
            pos_t = torch.tensor(static_pos, dtype=points.dtype, device=points.device)
            distances = torch.norm(points[:, :2] - pos_t[:2].unsqueeze(0), dim=-1)
            mask &= (distances > self.static_radius)

        return points[mask]

    def _cluster_points(self, points: torch.Tensor) -> list[torch.Tensor]:
        """Simple distance-based clustering (greedy).

        Args:
            points: 3D points, shape (N, 3).

        Returns:
            List of point clusters (each a tensor of shape (M, 3)).
        """
        if points.shape[0] == 0:
            return []

        clusters = []
        remaining = points.clone()
        assigned = torch.zeros(remaining.shape[0], dtype=torch.bool, device=points.device)

        while not assigned.all():
            # Pick first unassigned point as seed
            unassigned_idx = torch.where(~assigned)[0]
            if len(unassigned_idx) == 0:
                break

            seed_idx = unassigned_idx[0]
            seed = remaining[seed_idx]

            # Find all points within cluster_distance
            dists = torch.norm(remaining - seed.unsqueeze(0), dim=-1)
            cluster_mask = (dists < self.cluster_distance) & (~assigned)

            # Iteratively expand cluster
            prev_count = 0
            while cluster_mask.sum() > prev_count:
                prev_count = cluster_mask.sum().item()
                cluster_center = remaining[cluster_mask].mean(dim=0)
                dists_to_center = torch.norm(remaining - cluster_center.unsqueeze(0), dim=-1)
                cluster_mask = (dists_to_center < self.cluster_distance) & (~assigned)

            assigned |= cluster_mask
            cluster_points = remaining[cluster_mask]
            if cluster_points.shape[0] > 0:
                clusters.append(cluster_points)

        return clusters
