# Copyright (c) 2026, Multi-Agent Surveillance Project
# Coordinate transformation utilities for back-projecting camera/LiDAR detections.

from __future__ import annotations

import torch
import numpy as np


def quat_to_rot_matrix(quat: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to 3×3 rotation matrix.

    Args:
        quat: Quaternion as (w, x, y, z), shape (..., 4).

    Returns:
        Rotation matrix, shape (..., 3, 3).
    """
    if isinstance(quat, np.ndarray):
        quat = torch.from_numpy(quat).float()

    quat = quat / quat.norm(dim=-1, keepdim=True)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Rotation matrix from quaternion
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r02 = 2.0 * (x * z + w * y)
    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - w * x)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    rot = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)

    return rot


def pixel_to_camera(
    u: torch.Tensor,
    v: torch.Tensor,
    depth: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> torch.Tensor:
    """Back-project pixel coordinates + depth to 3D camera-frame coordinates.

    Uses pinhole model:
        X_cam = (u - cx) * depth / fx
        Y_cam = (v - cy) * depth / fy
        Z_cam = depth

    Args:
        u: Pixel x-coordinates, shape (N,).
        v: Pixel y-coordinates, shape (N,).
        depth: Depth values at each pixel, shape (N,).
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point in pixels.

    Returns:
        3D points in camera frame, shape (N, 3).
    """
    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    z_cam = depth

    return torch.stack([x_cam, y_cam, z_cam], dim=-1)


def camera_to_world(
    points_cam: torch.Tensor,
    cam_pos: torch.Tensor,
    cam_quat: torch.Tensor,
    convention: str = "world",
) -> torch.Tensor:
    """Transform 3D points from camera frame to world frame.

    The pinhole back-projection produces points in optical frame:
        X_cam = right, Y_cam = down, Z_cam = forward (depth)

    quat_w_world from Isaac Lab uses "world" body convention:
        Body: Forward = +X, Right = +Y, Up = +Z

    So we map: body_X = Z_cam, body_Y = X_cam, body_Z = -Y_cam

    Args:
        points_cam: Points in camera optical frame, shape (N, 3).
        cam_pos: Camera position in world, shape (3,).
        cam_quat: Camera orientation (w, x, y, z), shape (4,).
        convention: Kept for API compat.

    Returns:
        Points in world frame, shape (N, 3).
    """
    # Map from optical frame (X=right, Y=down, Z=depth) to
    # "world" body frame (X=forward, Y=right, Z=up)
    points_body = torch.zeros_like(points_cam)
    points_body[..., 0] = points_cam[..., 2]   # body_X = Z_cam (forward = depth)
    points_body[..., 1] = -points_cam[..., 0]  # body_Y = -X_cam (left = -right)
    points_body[..., 2] = -points_cam[..., 1]  # body_Z = -Y_cam (up = -down)

    # Rotate by camera orientation and translate
    rot_matrix = quat_to_rot_matrix(cam_quat)  # (3, 3)
    points_world = torch.matmul(points_body, rot_matrix.T) + cam_pos

    return points_world


def sensor_to_world(
    points_sensor: torch.Tensor,
    sensor_pos: torch.Tensor,
    sensor_quat: torch.Tensor,
) -> torch.Tensor:
    """Transform 3D points from sensor frame to world frame.

    Simple rotation + translation for LiDAR points.

    Args:
        points_sensor: Points in sensor frame, shape (N, 3).
        sensor_pos: Sensor position in world, shape (3,).
        sensor_quat: Sensor orientation (w, x, y, z) in world, shape (4,).

    Returns:
        Points in world frame, shape (N, 3).
    """
    rot_matrix = quat_to_rot_matrix(sensor_quat)
    points_world = torch.matmul(points_sensor, rot_matrix.T) + sensor_pos
    return points_world


def quat_apply(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate `vec` by quaternion(s) `quat_wxyz` (no Isaac dependency).

    Implements the standard `v + 2 w (q × v) + 2 (q × (q × v))` form, which
    matches `isaaclab.utils.math.quat_apply` numerically. Both inputs may
    have leading batch dims; quaternion shape is broadcast over the vector
    shape.

    Args:
        quat_wxyz: Quaternion in (w, x, y, z) order, shape (..., 4).
        vec:       3-vector(s) in the rotated frame, shape (..., 3).

    Returns:
        Rotated 3-vector(s), broadcast-shaped against `vec`.
    """
    if quat_wxyz.dim() == 1:
        quat_wxyz = quat_wxyz.unsqueeze(0)
    while quat_wxyz.dim() < vec.dim():
        quat_wxyz = quat_wxyz.unsqueeze(-2)
    w = quat_wxyz[..., 0:1]
    xyz = quat_wxyz[..., 1:4]
    xyz_b = xyz.expand_as(vec)
    t = 2.0 * torch.cross(xyz_b, vec, dim=-1)
    return vec + w * t + torch.cross(xyz_b, t, dim=-1)


def transform_points(
    points: torch.Tensor,
    pos: torch.Tensor,
    quat_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Apply SE(3) (rotate by `quat_wxyz`, then translate by `pos`) to `points`.

    Mirrors `isaaclab.utils.math.transform_points` for the typical
    perception use case: `points` is `[N, 3]`, `pos` is `[1, 3]`,
    `quat_wxyz` is `[1, 4]`, output is `[1, N, 3]`.

    Args:
        points:    Points in source frame, shape (..., N, 3).
        pos:       Translation, shape (..., 3).
        quat_wxyz: Source-to-target rotation, shape (..., 4).

    Returns:
        Transformed points with the broadcast-shaped batch dims.
    """
    if points.dim() == 2:
        points = points.unsqueeze(0)
    rotated = quat_apply(quat_wxyz, points)
    return rotated + pos.unsqueeze(-2)


def unproject_depth(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Back-project a depth image to 3D points in the camera optical frame.

    Pure-torch replacement for `isaaclab.utils.math.unproject_depth`. The
    optical frame matches ROS / OpenCV: +X right, +Y down, +Z forward
    (along the optical axis).

    Args:
        depth:      Depth image with finite values for valid pixels and
                    +inf (or non-finite) where no depth is known. Shape
                    (..., H, W).
        intrinsics: 3×3 pinhole intrinsic matrix.

    Returns:
        Back-projected points, shape (..., H, W, 3). Pixels whose depth
        is non-finite are returned as +inf so callers can mask them out
        with `torch.isfinite(...).all(dim=-1)`.
    """
    if depth.dim() == 2:
        depth_h, depth_w = depth.shape
    else:
        depth_h, depth_w = depth.shape[-2:]
    device = depth.device
    dtype = depth.dtype

    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]

    u = torch.arange(depth_w, device=device, dtype=dtype)
    v = torch.arange(depth_h, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(v, u, indexing="ij")

    z = depth
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    return torch.stack([x, y, z], dim=-1)


def compute_position_error(
    estimated: torch.Tensor,
    ground_truth: torch.Tensor,
) -> float:
    """Compute Euclidean distance error between estimated and ground truth positions.

    Args:
        estimated: Estimated position, shape (3,).
        ground_truth: Ground truth position, shape (3,).

    Returns:
        Euclidean distance error in meters.
    """
    if isinstance(estimated, np.ndarray):
        estimated = torch.from_numpy(estimated).float()
    if isinstance(ground_truth, np.ndarray):
        ground_truth = torch.from_numpy(ground_truth).float()
    return torch.norm(estimated - ground_truth).item()
