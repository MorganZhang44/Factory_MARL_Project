"""Lightweight 2D static-map scan matching for dog self-localization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from environment.static_scene_geometry import WALL_HEIGHT, WALL_THICKNESS, get_static_cuboids, get_static_cylinders


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class StaticMap2D:
    """Dense 2D point representation of the static scene layout."""

    points_xy: np.ndarray
    spacing: float = 0.25
    metadata: dict[str, Any] | None = None


@dataclass
class MatchResult:
    """Result of 2D scan-to-map matching."""

    pose_xyyaw: tuple[float, float, float]
    score: float
    inliers: int
    rmse: float
    accepted: bool
    xy_correction: tuple[float, float]
    yaw_correction: float
    used_prediction_only: bool


def _as_numpy(value: torch.Tensor | np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


def _quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1e-9:
        return np.eye(3)
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _yaw_rotation(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def transform_points_2d(points_xy: np.ndarray, pose_xyyaw: tuple[float, float, float]) -> np.ndarray:
    """Transform local XY points into world XY with an SE(2) pose."""
    pose = np.asarray(pose_xyyaw, dtype=np.float64)
    rot = _yaw_rotation(float(pose[2]))
    return points_xy @ rot.T + pose[:2]


def world_hits_to_local_points(
    hit_points_w: torch.Tensor | np.ndarray,
    sensor_pos_w: torch.Tensor | np.ndarray,
    sensor_quat_w: torch.Tensor | np.ndarray,
) -> np.ndarray:
    """Convert world-frame LiDAR hit points back into the sensor frame."""
    hits = _as_numpy(hit_points_w)
    if hits.ndim == 3:
        hits = hits[0]

    pos = _as_numpy(sensor_pos_w)
    if pos.ndim == 2:
        pos = pos[0]

    quat = _as_numpy(sensor_quat_w)
    if quat.ndim == 2:
        quat = quat[0]

    valid = np.isfinite(hits).all(axis=1)
    hits = hits[valid]
    if hits.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    rot_wb = _quat_wxyz_to_rotmat(quat)
    rot_bw = rot_wb.T
    return (hits - pos.reshape(1, 3)) @ rot_bw.T


def downsample_xy(points_xy: np.ndarray, voxel_size: float = 0.20, max_points: int = 450) -> np.ndarray:
    """Voxel downsample 2D points while preserving spatial spread."""
    if len(points_xy) == 0:
        return points_xy.reshape(0, 2)

    quantized = np.floor(points_xy / max(voxel_size, 1.0e-6)).astype(np.int32)
    _, unique_idx = np.unique(quantized, axis=0, return_index=True)
    downsampled = points_xy[np.sort(unique_idx)]

    if len(downsampled) <= max_points:
        return downsampled

    stride = max(1, len(downsampled) // max_points)
    return downsampled[::stride][:max_points]


def filter_local_scan_for_matching(
    local_points: np.ndarray,
    min_range: float = 0.60,
    max_range: float = 25.0,
    z_min: float = -0.60,
    z_max: float = 0.90,
    voxel_size: float = 0.20,
    max_points: int = 450,
) -> np.ndarray:
    """Keep obstacle-like LiDAR returns and project them into XY."""
    if len(local_points) == 0:
        return np.empty((0, 2), dtype=np.float64)

    ranges = np.linalg.norm(local_points[:, :2], axis=1)
    mask = np.isfinite(local_points).all(axis=1)
    mask &= ranges >= min_range
    mask &= ranges <= max_range
    mask &= local_points[:, 2] >= z_min
    mask &= local_points[:, 2] <= z_max

    scan_xy = local_points[mask, :2]
    if len(scan_xy) == 0:
        return np.empty((0, 2), dtype=np.float64)

    return downsample_xy(scan_xy, voxel_size=voxel_size, max_points=max_points)


def _sample_segment(start_xy: tuple[float, float], end_xy: tuple[float, float], spacing: float) -> np.ndarray:
    start = np.asarray(start_xy, dtype=np.float64)
    end = np.asarray(end_xy, dtype=np.float64)
    dist = np.linalg.norm(end - start)
    num = max(2, int(math.ceil(dist / spacing)) + 1)
    alphas = np.linspace(0.0, 1.0, num=num)
    return start[None, :] + (end - start)[None, :] * alphas[:, None]


def _sample_box_outline(center_xy: tuple[float, float], size_xy: tuple[float, float], spacing: float) -> np.ndarray:
    cx, cy = center_xy
    sx, sy = size_xy
    hx = 0.5 * sx
    hy = 0.5 * sy
    corners = [
        (cx - hx, cy - hy),
        (cx + hx, cy - hy),
        (cx + hx, cy + hy),
        (cx - hx, cy + hy),
    ]
    segments = []
    for i in range(4):
        segments.append(_sample_segment(corners[i], corners[(i + 1) % 4], spacing))
    return np.vstack(segments)


def _sample_circle_outline(center_xy: tuple[float, float], radius: float, spacing: float) -> np.ndarray:
    circumference = 2.0 * math.pi * radius
    num = max(12, int(math.ceil(circumference / spacing)))
    theta = np.linspace(0.0, 2.0 * math.pi, num=num, endpoint=False)
    cx, cy = center_xy
    points = np.zeros((num, 2), dtype=np.float64)
    points[:, 0] = cx + radius * np.cos(theta)
    points[:, 1] = cy + radius * np.sin(theta)
    return points


def build_static_map_from_scene_cfg(point_spacing: float = 0.25) -> StaticMap2D:
    """Build a dense 2D point map that mirrors the current static scene layout."""
    points = []
    cuboids = get_static_cuboids()
    cylinders = get_static_cylinders()
    wall_count = 0
    box_count = 0
    for cuboid in cuboids:
        center = cuboid["center"]
        size = cuboid["size"]
        points.append(_sample_box_outline(center[:2], size[:2], point_spacing))
        if cuboid["label"] == "wall":
            wall_count += 1
        elif cuboid["label"] == "box":
            box_count += 1
    for cylinder in cylinders:
        points.append(_sample_circle_outline(cylinder["center"][:2], cylinder["radius"], point_spacing))

    map_points = np.vstack(points)
    map_points = downsample_xy(map_points, voxel_size=point_spacing * 0.5, max_points=5000)
    metadata = {
        "num_points": int(len(map_points)),
        "wall_segments": wall_count,
        "pillars": len(cylinders),
        "boxes": box_count,
        "wall_thickness": WALL_THICKNESS,
        "wall_height": WALL_HEIGHT,
    }
    return StaticMap2D(points_xy=map_points, spacing=point_spacing, metadata=metadata)


def _estimate_rigid_transform_2d(source_xy: np.ndarray, target_xy: np.ndarray) -> tuple[float, float, float]:
    """Estimate SE(2) pose that maps source points into target points."""
    src_centroid = source_xy.mean(axis=0)
    tgt_centroid = target_xy.mean(axis=0)
    src_centered = source_xy - src_centroid
    tgt_centered = target_xy - tgt_centroid

    cov = src_centered.T @ tgt_centered
    u, _, vt = np.linalg.svd(cov)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0.0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T

    trans = tgt_centroid - rot @ src_centroid
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    return float(trans[0]), float(trans[1]), float(wrap_angle(yaw))


def match_scan(
    pred_pose_xyyaw: tuple[float, float, float],
    scan_xy: np.ndarray,
    static_map: StaticMap2D,
    max_iterations: int = 6,
    trim_ratio: float = 0.75,
    max_correspondence_dist: float = 1.25,
    min_inliers: int = 20,
    max_translation_delta: float = 1.0,
    max_yaw_delta: float = math.radians(25.0),
) -> MatchResult:
    """Match a local LiDAR scan to the static 2D map using trimmed ICP."""
    pred = np.asarray(pred_pose_xyyaw, dtype=np.float64)
    scan_xy = np.asarray(scan_xy, dtype=np.float64).reshape(-1, 2)
    map_xy = static_map.points_xy
    required_inliers = max(6, min(int(min_inliers), len(scan_xy)))

    if len(scan_xy) < required_inliers or len(map_xy) == 0:
        return MatchResult(
            pose_xyyaw=(float(pred[0]), float(pred[1]), float(pred[2])),
            score=0.0,
            inliers=0,
            rmse=float("inf"),
            accepted=False,
            xy_correction=(0.0, 0.0),
            yaw_correction=0.0,
            used_prediction_only=True,
        )

    pose = pred.copy()
    inlier_count = 0
    rmse = float("inf")

    for _ in range(max_iterations):
        scan_world = transform_points_2d(scan_xy, (float(pose[0]), float(pose[1]), float(pose[2])))
        distances = np.linalg.norm(scan_world[:, None, :] - map_xy[None, :, :], axis=2)
        nearest_idx = np.argmin(distances, axis=1)
        nearest_dist = distances[np.arange(len(scan_world)), nearest_idx]

        valid_idx = np.where(nearest_dist < max_correspondence_dist)[0]
        if len(valid_idx) < required_inliers:
            break

        keep_count = max(required_inliers, int(len(valid_idx) * trim_ratio))
        best_order = np.argsort(nearest_dist[valid_idx])[:keep_count]
        inlier_scan_idx = valid_idx[best_order]
        matched_map = map_xy[nearest_idx[inlier_scan_idx]]

        est_pose = np.asarray(_estimate_rigid_transform_2d(scan_xy[inlier_scan_idx], matched_map), dtype=np.float64)
        delta_xy = np.clip(est_pose[:2] - pred[:2], -max_translation_delta, max_translation_delta)
        delta_yaw = np.clip(wrap_angle(float(est_pose[2] - pred[2])), -max_yaw_delta, max_yaw_delta)
        new_pose = np.array([pred[0] + delta_xy[0], pred[1] + delta_xy[1], wrap_angle(float(pred[2] + delta_yaw))])

        step_delta = np.array(
            [
                new_pose[0] - pose[0],
                new_pose[1] - pose[1],
                wrap_angle(float(new_pose[2] - pose[2])),
            ],
            dtype=np.float64,
        )
        pose = new_pose
        inlier_count = len(inlier_scan_idx)
        rmse = float(np.sqrt(np.mean(nearest_dist[inlier_scan_idx] ** 2)))

        if np.linalg.norm(step_delta[:2]) < 1.0e-3 and abs(step_delta[2]) < math.radians(0.1):
            break

    scan_world = transform_points_2d(scan_xy, (float(pose[0]), float(pose[1]), float(pose[2])))
    distances = np.linalg.norm(scan_world[:, None, :] - map_xy[None, :, :], axis=2)
    nearest_idx = np.argmin(distances, axis=1)
    nearest_dist = distances[np.arange(len(scan_world)), nearest_idx]
    valid_mask = nearest_dist < max_correspondence_dist
    inlier_count = int(valid_mask.sum())
    rmse = float(np.sqrt(np.mean(nearest_dist[valid_mask] ** 2))) if inlier_count > 0 else float("inf")

    correction_xy = pose[:2] - pred[:2]
    correction_yaw = wrap_angle(float(pose[2] - pred[2]))
    score = 0.0
    if inlier_count > 0 and math.isfinite(rmse):
        coverage = min(1.0, inlier_count / max(float(len(scan_xy)), 1.0))
        fit = 1.0 / (1.0 + rmse)
        score = float(coverage * fit)

    accepted = (
        inlier_count >= required_inliers
        and math.isfinite(rmse)
        and rmse < 0.75
        and np.linalg.norm(correction_xy) <= max_translation_delta + 1.0e-6
        and abs(correction_yaw) <= max_yaw_delta + 1.0e-6
    )
    return MatchResult(
        pose_xyyaw=(float(pose[0]), float(pose[1]), float(pose[2])),
        score=score,
        inliers=inlier_count,
        rmse=rmse,
        accepted=accepted,
        xy_correction=(float(correction_xy[0]), float(correction_xy[1])),
        yaw_correction=float(correction_yaw),
        used_prediction_only=not accepted,
    )
