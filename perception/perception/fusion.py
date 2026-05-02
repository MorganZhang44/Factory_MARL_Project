# Copyright (c) 2026, Multi-Agent Surveillance Project
# Multi-sensor fusion for suspect localization.

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .camera_detector import CameraDetection
from .lidar_detector import LidarDetection
from environment.static_scene_geometry import is_point_walkable


@dataclass
class FusionResult:
    """Result of multi-sensor fusion for suspect localization.

    Both `position_world` and `velocity_world` are expressed in the global
    world frame (same convention as Isaac Lab's `root_pos_w` / `root_lin_vel_w`).
    XY comes from the 2D Kalman filter; Z is tracked separately as an EMA of
    depth-bearing camera and LiDAR observations. Velocity Z is held at 0
    because the tracker is planar.
    """
    timestamp: float = 0.0
    step: int = 0
    detected: bool = False
    position_world: Optional[torch.Tensor] = None     # (3,) fused world position
    velocity_world: Optional[torch.Tensor] = None     # (3,) fused world velocity
    confidence: float = 0.0
    ground_truth: Optional[torch.Tensor] = None       # (3,) for validation
    error_meters: float = float("inf")
    num_camera_detections: int = 0
    num_lidar_detections: int = 0
    camera_detections: list[CameraDetection] = field(default_factory=list)
    lidar_detections: list[LidarDetection] = field(default_factory=list)


class SensorFusion:
    """Fuse detections from multiple cameras and LiDARs to estimate suspect position.

    Uses a 2D Kalman filter (state = [x, y, vx, vy]) for temporal smoothing
    instead of simple EMA. This provides:
    - Motion prediction during missed detections
    - Adaptive measurement weighting based on uncertainty
    - Velocity estimation

    Fusion strategy:
    1. Collect all valid detections (position + confidence) from cameras and LiDARs
    2. Weight each detection by its confidence score
    3. Compute weighted average position as measurement
    4. Update Kalman filter with measurement
    """

    def __init__(
        self,
        camera_weight: float = 1.0,
        lidar_weight: float = 1.2,
        temporal_alpha: float = 0.7,  # kept for API compat but unused
        outlier_threshold: float = 5.0,
        history_size: int = 100,
        process_noise: float = 1.0,      # Lower q = smoother velocity, slight lag
        measurement_noise: float = 2.0,  # Lower = trust sensors more, less lag
        dt: float = 0.083,              # 5-step interval at 60Hz
        camera_gate_distance: float = 2.0,
        lidar_gate_distance: float = 3.0,
    ):
        self.camera_weight = camera_weight
        self.lidar_weight = lidar_weight
        self.outlier_threshold = outlier_threshold
        self.history_size = history_size
        self.camera_gate_distance = camera_gate_distance
        self.lidar_gate_distance = lidar_gate_distance

        self._history: list[FusionResult] = []

        # --- Kalman filter state: [x, y, vx, vy] ---
        self._kf_initialized = False
        self._dt = dt

        # State vector: [x, y, vx, vy]
        self._x = np.zeros(4)

        # State covariance (start with high uncertainty)
        self._P = np.eye(4) * 100.0

        # State transition: constant velocity model
        self._F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        # Measurement matrix: we observe [x, y]
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        # Process noise
        q = process_noise
        self._Q = np.array([
            [q*dt**2, 0, q*dt, 0],
            [0, q*dt**2, 0, q*dt],
            [q*dt, 0, q, 0],
            [0, q*dt, 0, q],
        ])

        # Measurement noise
        self._R = np.eye(2) * measurement_noise**2
        self._last_camera_ray_support = 0
        self._last_camera_ray_weight = 0.0
        self._last_camera_fallback_support = 0
        self._last_camera_fallback_weight = 0.0

        # Tracked world-frame Z for the intruder (KF tracks XY only).
        # Updated from depth-bearing camera detections and LiDAR cluster centroids
        # so the published 3D pose is in the world frame, not pinned to a constant.
        self._z_default = 0.9
        self._z_estimate: float | None = None

        # Output-side velocity rate-limiter: keeps a single noisy CCTV foot
        # point from spiking the published velocity by >1 m/s, without adding
        # the lag of an EMA. The internal KF state stays untouched.
        self._smoothed_vel_xy: np.ndarray | None = None
        self._velocity_max_change = 0.22  # m/s allowed between consecutive fuses
        self._intruder_max_speed = 1.10   # m/s safety clip (humans don't sprint here)

    def fuse(
        self,
        camera_detections: list[CameraDetection],
        lidar_detections: list[LidarDetection],
        ground_truth: Optional[torch.Tensor] = None,
        timestamp: float = 0.0,
        step: int = 0,
    ) -> FusionResult:
        """Fuse all sensor detections into a single position estimate.

        Args:
            camera_detections: List of camera detection results.
            lidar_detections: List of LiDAR detection results.
            ground_truth: Optional ground truth position for error computation.
            timestamp: Current simulation time.
            step: Current simulation step.

        Returns:
            FusionResult with fused position estimate.
        """
        # --- Kalman predict step (always) ---
        self._kf_predict()
        predicted_xy = self._get_predicted_xy()

        num_cam = sum(1 for d in camera_detections if d.detected)
        num_lid = sum(1 for d in lidar_detections if d.detected)

        # Update the world-frame Z estimate from depth-bearing detections so the
        # published 3D position is not pinned to a constant body-center height.
        self._update_z_estimate(camera_detections, lidar_detections)

        camera_positions, camera_weights = self._collect_camera_measurements(camera_detections, predicted_xy)
        lidar_positions, lidar_weights = self._collect_lidar_measurements(lidar_detections, predicted_xy)

        device = None
        if camera_positions:
            device = camera_positions[0].device
        elif lidar_positions:
            device = lidar_positions[0].device

        camera_consensus, camera_support, camera_total_weight = self._consolidate_measurements(
            camera_positions,
            camera_weights,
            predicted_xy,
            radius=self.camera_gate_distance,
        )
        lidar_consensus, lidar_support, lidar_total_weight = self._consolidate_measurements(
            lidar_positions,
            lidar_weights,
            predicted_xy,
            radius=self.lidar_gate_distance,
        )

        if camera_consensus is None and lidar_consensus is not None and predicted_xy is not None:
            pred = torch.tensor(predicted_xy, dtype=lidar_consensus.dtype, device=lidar_consensus.device)
            lidar_pred_dist = torch.norm(lidar_consensus[:2] - pred).item()
            if lidar_pred_dist > max(0.9, 0.45 * self.lidar_gate_distance):
                lidar_consensus = None
                lidar_support = 0
                lidar_total_weight = 0.0

        if camera_consensus is None and lidar_consensus is None:
            # No detections — use predicted state
            kf_pos = self._get_kf_position(device or "cpu")
            kf_vel = self._get_kf_velocity(device or "cpu")
            result = FusionResult(
                timestamp=timestamp,
                step=step,
                detected=False,
                position_world=kf_pos if self._kf_initialized else None,
                velocity_world=kf_vel if self._kf_initialized else None,
                confidence=0.0,
                ground_truth=ground_truth,
                error_meters=self._compute_error(kf_pos if self._kf_initialized else None, ground_truth),
                num_camera_detections=num_cam,
                num_lidar_detections=num_lid,
                camera_detections=camera_detections,
                lidar_detections=lidar_detections,
            )
            self._add_to_history(result)
            return result

        if (not self._kf_initialized) and camera_consensus is not None and camera_support >= 2:
            measurement = camera_consensus
            modality_weight_sum = camera_total_weight
        else:
            modality_positions: list[torch.Tensor] = []
            modality_weights: list[float] = []

            if camera_consensus is not None:
                modality_positions.append(camera_consensus)
                modality_weights.append(camera_total_weight)

            if lidar_consensus is not None:
                adjusted_lidar_weight = self._resolve_modality_disagreement(
                    camera_consensus,
                    camera_support,
                    camera_total_weight,
                    lidar_consensus,
                    lidar_total_weight,
                    predicted_xy,
                )
                if adjusted_lidar_weight > 1.0e-4:
                    modality_positions.append(lidar_consensus)
                    modality_weights.append(adjusted_lidar_weight)

            if not modality_positions:
                kf_pos = self._get_kf_position(device or "cpu")
                kf_vel = self._get_kf_velocity(device or "cpu")
                result = FusionResult(
                    timestamp=timestamp,
                    step=step,
                    detected=False,
                    position_world=kf_pos if self._kf_initialized else None,
                    velocity_world=kf_vel if self._kf_initialized else None,
                    confidence=0.0,
                    ground_truth=ground_truth,
                    error_meters=self._compute_error(kf_pos if self._kf_initialized else None, ground_truth),
                    num_camera_detections=num_cam,
                    num_lidar_detections=num_lid,
                    camera_detections=camera_detections,
                    lidar_detections=lidar_detections,
                )
                self._add_to_history(result)
                return result

            if len(modality_positions) == 1:
                measurement = modality_positions[0]
                modality_weight_sum = modality_weights[0]
            else:
                positions = torch.stack(modality_positions)
                w = torch.tensor(modality_weights, dtype=torch.float32, device=positions.device)
                w_norm = w / w.sum()
                measurement = (positions * w_norm.unsqueeze(-1)).sum(dim=0)
                modality_weight_sum = float(w.sum().item())

        # Overall confidence
        total_confidence = min(1.0, modality_weight_sum / max(1, len(camera_detections) + len(lidar_detections)))

        # --- Kalman update step ---
        meas_xy = measurement[:2].cpu().numpy()
        if self._should_reacquire_from_camera(meas_xy, camera_consensus, camera_support, camera_total_weight):
            self._kf_reinitialize(meas_xy)
        else:
            self._kf_update(meas_xy)

        # Get filtered position
        fused_pos = self._get_kf_position(device)
        fused_vel = self._get_kf_velocity(device)

        # Error vs ground truth
        error = self._compute_error(fused_pos, ground_truth)

        result = FusionResult(
            timestamp=timestamp,
            step=step,
            detected=True,
            position_world=fused_pos,
            velocity_world=fused_vel,
            confidence=total_confidence,
            ground_truth=ground_truth,
            error_meters=error,
            num_camera_detections=num_cam,
            num_lidar_detections=num_lid,
            camera_detections=camera_detections,
            lidar_detections=lidar_detections,
        )

        self._add_to_history(result)
        return result

    def get_history(self) -> list[FusionResult]:
        """Return the history of fusion results."""
        return self._history.copy()

    def reset(self):
        """Reset the fusion state including Kalman filter."""
        self._kf_initialized = False
        self._x = np.zeros(4)
        self._P = np.eye(4) * 100.0
        self._last_camera_ray_support = 0
        self._last_camera_ray_weight = 0.0
        self._last_camera_fallback_support = 0
        self._last_camera_fallback_weight = 0.0
        self._z_estimate = None
        self._smoothed_vel_xy = None
        self._history.clear()

    # ------------------------------------------------------------------
    # Kalman filter helpers
    # ------------------------------------------------------------------

    def _kf_predict(self):
        """Kalman predict step: advance state by one time step."""
        if not self._kf_initialized:
            return
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q

    def _kf_update(self, measurement: np.ndarray):
        """Kalman update step: correct state with measurement [x, y].

        Args:
            measurement: np.ndarray of shape (2,), observed [x, y].
        """
        if not self._kf_initialized:
            # Initialize state from first measurement
            self._x[0] = measurement[0]
            self._x[1] = measurement[1]
            self._x[2] = 0.0  # unknown velocity
            self._x[3] = 0.0
            self._P = np.eye(4) * 10.0  # moderate initial uncertainty
            self._kf_initialized = True
            return

        # Innovation
        z = measurement
        y = z - self._H @ self._x  # (2,)

        # Innovation covariance
        S = self._H @ self._P @ self._H.T + self._R  # (2, 2)

        # Kalman gain
        K = self._P @ self._H.T @ np.linalg.inv(S)  # (4, 2)

        # State update
        self._x = self._x + K @ y

        # Covariance update (Joseph form for stability)
        I_KH = np.eye(4) - K @ self._H
        self._P = I_KH @ self._P @ I_KH.T + K @ self._R @ K.T

    def _kf_reinitialize(self, measurement: np.ndarray):
        """Hard-reset the track when strong multi-camera geometry contradicts drift."""
        self._x[0] = measurement[0]
        self._x[1] = measurement[1]
        self._x[2] = 0.0
        self._x[3] = 0.0
        self._P = np.eye(4) * 4.0
        self._kf_initialized = True

    def _should_reacquire_from_camera(
        self,
        measurement_xy: np.ndarray,
        camera_consensus: torch.Tensor | None,
        camera_support: int,
        camera_weight_sum: float,
    ) -> bool:
        """Allow strong multi-CCTV consensus to break out of a stale prediction."""
        raw_support = max(camera_support, self._last_camera_ray_support, self._last_camera_fallback_support)
        raw_weight = max(camera_weight_sum, self._last_camera_ray_weight, self._last_camera_fallback_weight)
        if not self._kf_initialized or camera_consensus is None or raw_support < 4:
            return False
        innovation = float(np.linalg.norm(measurement_xy - self._x[:2]))
        velocity = float(np.linalg.norm(self._x[2:]))
        reacquire_gate = max(3.0, 2.0 * self.camera_gate_distance + 0.5 * velocity)
        return innovation > reacquire_gate and raw_weight > 0.15

    def _get_kf_position(self, device) -> torch.Tensor:
        """Get current Kalman filter position in the world frame.

        XY comes from the 2D Kalman filter and Z from `_z_estimate`, an EMA of
        the world-frame Z observed by depth-bearing camera detections and
        LiDAR cluster centroids. When no Z observations have arrived yet, Z
        falls back to `_z_default` (approximate humanoid body-center height).
        """
        z_value = self._z_estimate if self._z_estimate is not None else self._z_default
        pos = torch.tensor(
            [self._x[0], self._x[1], z_value],
            dtype=torch.float32,
        )
        if device is not None and device != "cpu":
            pos = pos.to(device)
        return pos

    def _get_kf_velocity(self, device) -> torch.Tensor:
        """Return the published world-frame velocity (rate-limited only).

        Raw KF velocity is rate-limited between consecutive fuses so a single
        noisy CCTV foot-point can't spike the published velocity by >1 m/s.
        No EMA — sustained changes pass through within ~3 fuses (~0.25 s).
        The internal KF state stays untouched for prediction/gating.
        """
        raw = np.array([self._x[2], self._x[3]], dtype=np.float64)

        raw_speed = float(np.linalg.norm(raw))
        if raw_speed > self._intruder_max_speed:
            raw = raw * (self._intruder_max_speed / max(raw_speed, 1.0e-6))

        if self._smoothed_vel_xy is None:
            self._smoothed_vel_xy = raw.copy()
        else:
            delta = raw - self._smoothed_vel_xy
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > self._velocity_max_change:
                delta = delta * (self._velocity_max_change / max(delta_norm, 1.0e-6))
            self._smoothed_vel_xy = self._smoothed_vel_xy + delta

        vel = torch.tensor(
            [self._smoothed_vel_xy[0], self._smoothed_vel_xy[1], 0.0],
            dtype=torch.float32,
        )
        if device is not None and device != "cpu":
            vel = vel.to(device)
        return vel

    def _get_predicted_xy(self) -> np.ndarray | None:
        """Return the current predicted XY state after the predict step."""
        if not self._kf_initialized:
            return None
        return self._x[:2].copy()

    def _update_z_estimate(
        self,
        camera_detections: list[CameraDetection],
        lidar_detections: list[LidarDetection],
    ) -> None:
        """Blend valid world-frame Z observations into a smoothed Z estimate.

        Camera back-projections that used real depth and LiDAR cluster
        centroids report a meaningful world-frame Z. Monocular CCTV foot/body
        projections are ignored here because their Z is synthesized.
        """
        z_values: list[tuple[float, float]] = []  # (z, weight)

        for det in camera_detections:
            if not det.detected or det.position_world is None:
                continue
            if not getattr(det, "used_depth", False):
                continue
            z = float(det.position_world[2].item())
            if not (0.05 <= z <= 2.5):
                continue
            z_values.append((z, max(float(det.confidence), 0.05)))

        for det in lidar_detections:
            if not det.detected or det.position_world is None:
                continue
            z = float(det.position_world[2].item())
            if not (0.05 <= z <= 2.5):
                continue
            z_values.append((z, 0.5 * max(float(det.confidence), 0.05)))

        if not z_values:
            return

        weight_sum = sum(w for _, w in z_values)
        if weight_sum <= 0.0:
            return
        observed_z = sum(z * w for z, w in z_values) / weight_sum

        if self._z_estimate is None:
            self._z_estimate = observed_z
        else:
            self._z_estimate = 0.7 * self._z_estimate + 0.3 * observed_z

    def _collect_camera_measurements(
        self,
        camera_detections: list[CameraDetection],
        predicted_xy: np.ndarray | None,
    ) -> tuple[list[torch.Tensor], list[float]]:
        positions: list[torch.Tensor] = []
        weights: list[float] = []
        self._last_camera_ray_support = 0
        self._last_camera_ray_weight = 0.0
        self._last_camera_fallback_support = 0
        self._last_camera_fallback_weight = 0.0

        ray_consensus, ray_support, ray_weight = self._triangulate_camera_rays(camera_detections, predicted_xy)
        use_ray_consensus = ray_consensus is not None and ray_support >= 2
        if use_ray_consensus:
            positions.append(ray_consensus)
            weights.append(ray_weight)
            self._last_camera_ray_support = ray_support
            self._last_camera_ray_weight = ray_weight

        for det in camera_detections:
            if not det.detected or det.position_world is None:
                continue
            if "dog" in det.camera_name.lower():
                continue
            if not self._is_plausible_fixed_camera_detection(det, predicted_xy):
                continue
            is_monocular = not getattr(det, "used_depth", False)
            det_weight = det.confidence * self.camera_weight
            if is_monocular:
                det_weight *= 0.35 * max(0.2, 1.0 - 0.6 * getattr(det, "occlusion_score", 0.0))
            det_weight *= self._prediction_gate(
                det.position_world[:2],
                predicted_xy,
                base_distance=self.camera_gate_distance,
                confidence=det.confidence,
                occlusion_score=getattr(det, "occlusion_score", 0.0),
            )
            if det_weight <= 1.0e-4:
                continue
            positions.append(det.position_world)
            weights.append(det_weight)

        if not positions:
            fallback, fallback_support, fallback_weight = self._ungated_camera_point_consensus(camera_detections)
            if fallback is not None and fallback_support >= 3:
                positions.append(fallback)
                weights.append(fallback_weight)
                self._last_camera_fallback_support = fallback_support
                self._last_camera_fallback_weight = fallback_weight
        return positions, weights

    def _ungated_camera_point_consensus(
        self,
        camera_detections: list[CameraDetection],
    ) -> tuple[torch.Tensor | None, int, float]:
        """Fallback re-acquisition from multiple fixed CCTV ground-point estimates."""
        positions: list[torch.Tensor] = []
        weights: list[float] = []
        for det in camera_detections:
            if not det.detected or det.position_world is None:
                continue
            if getattr(det, "used_depth", False) or "dog" in det.camera_name.lower():
                continue
            if not self._is_plausible_fixed_camera_detection(det, predicted_xy=None):
                continue
            confidence = float(det.confidence)
            visibility = float(getattr(det, "visibility_score", 1.0))
            occlusion = float(getattr(det, "occlusion_score", 0.0))
            weight = confidence * self.camera_weight * max(0.2, visibility) * max(0.15, 1.0 - 0.55 * occlusion)
            if weight <= 1.0e-4:
                continue
            positions.append(det.position_world)
            weights.append(weight)

        if len(positions) < 3:
            return None, len(positions), 0.0

        pos_t = torch.stack(positions)
        w_t = torch.tensor(weights, dtype=torch.float32, device=pos_t.device)
        median = pos_t[:, :2].median(dim=0).values
        dists = torch.norm(pos_t[:, :2] - median.unsqueeze(0), dim=-1)
        keep = dists <= max(1.6, float(dists.median().item()) * 2.5 + 0.2)
        if keep.sum() < 3:
            return None, int(keep.sum().item()), 0.0

        pos_t = pos_t[keep]
        w_t = w_t[keep]
        w_norm = w_t / w_t.sum()
        consensus = (pos_t * w_norm.unsqueeze(-1)).sum(dim=0)
        consensus[2] = 0.9
        return consensus, int(pos_t.shape[0]), float(w_t.sum().item())

    def _triangulate_camera_rays(
        self,
        camera_detections: list[CameraDetection],
        predicted_xy: np.ndarray | None,
    ) -> tuple[torch.Tensor | None, int, float]:
        """Estimate target XY from multiple monocular CCTV bearing rays."""
        rays: list[tuple[torch.Tensor, torch.Tensor, float]] = []
        for det in camera_detections:
            if not det.detected or getattr(det, "used_depth", False):
                continue
            if "dog" in det.camera_name.lower():
                continue
            if not self._is_plausible_fixed_camera_detection(det, predicted_xy):
                continue
            origin = getattr(det, "ray_origin_world", None)
            direction = getattr(det, "ray_dir_world", None)
            if origin is None or direction is None:
                continue
            dir_xy = direction[:2]
            dir_norm = torch.norm(dir_xy).item()
            if dir_norm < 1.0e-4:
                continue
            dir_xy = dir_xy / dir_norm
            conf = float(det.confidence)
            visibility = float(getattr(det, "visibility_score", 1.0))
            occlusion = float(getattr(det, "occlusion_score", 0.0))
            weight = conf * self.camera_weight * max(0.15, visibility) * max(0.2, 1.0 - 0.5 * occlusion)
            if weight <= 1.0e-4:
                continue
            rays.append((origin[:2], dir_xy, weight))

        if len(rays) < 2:
            return None, len(rays), 0.0

        device = rays[0][0].device
        dtype = rays[0][0].dtype

        def solve(active_rays: list[tuple[torch.Tensor, torch.Tensor, float]]) -> torch.Tensor | None:
            a = torch.zeros((2, 2), dtype=dtype, device=device)
            b = torch.zeros(2, dtype=dtype, device=device)
            for origin, direction, weight in active_rays:
                normal = torch.stack((-direction[1], direction[0]))
                nn = normal.unsqueeze(1) @ normal.unsqueeze(0)
                a += weight * nn
                b += weight * (nn @ origin)
            det_a = torch.det(a).item()
            if abs(det_a) < 1.0e-5:
                return None
            return torch.linalg.solve(a, b)

        active = rays
        xy = solve(active)
        if xy is None:
            return None, len(rays), 0.0

        # Trim rays that are clearly inconsistent with the line intersection.
        for _ in range(2):
            residuals = []
            front_mask = []
            for origin, direction, _weight in active:
                normal = torch.stack((-direction[1], direction[0]))
                residuals.append(abs(torch.dot(normal, xy - origin)).item())
                front_mask.append(torch.dot(xy - origin, direction).item() > -0.5)
            if len(residuals) < 3:
                break
            med = float(np.median(residuals))
            cutoff = max(0.55, 2.5 * med + 0.15)
            trimmed = [ray for ray, res, in_front in zip(active, residuals, front_mask) if res <= cutoff and in_front]
            if len(trimmed) == len(active) or len(trimmed) < 2:
                break
            new_xy = solve(trimmed)
            if new_xy is None:
                break
            active = trimmed
            xy = new_xy

        residuals = []
        weight_sum = 0.0
        for origin, direction, weight in active:
            normal = torch.stack((-direction[1], direction[0]))
            residuals.append(abs(torch.dot(normal, xy - origin)).item())
            weight_sum += weight
        mean_residual = float(np.mean(residuals)) if residuals else float("inf")

        point_consistency_dist = float("inf")
        point_positions = [
            det.position_world[:2]
            for det in camera_detections
            if det.detected and (not getattr(det, "used_depth", False)) and det.position_world is not None
        ]
        if len(point_positions) >= 2:
            point_xy = torch.stack(point_positions).median(dim=0).values
            point_consistency_dist = torch.norm(xy - point_xy).item()
            if point_consistency_dist > max(2.6, self.camera_gate_distance + 0.8):
                return None, len(active), 0.0

        if predicted_xy is not None:
            pred = torch.tensor(predicted_xy, dtype=dtype, device=device)
            pred_dist = torch.norm(xy - pred).item()
            if pred_dist > max(1.6, self.camera_gate_distance + 0.6 * float(np.linalg.norm(self._x[2:]))):
                return None, 0, 0.0

        if mean_residual > 1.1:
            return None, len(active), 0.0

        pos = torch.tensor([xy[0].item(), xy[1].item(), 0.9], dtype=dtype, device=device)
        residual_quality = max(0.2, 1.0 - mean_residual / 1.2)
        return pos, len(active), weight_sum * residual_quality

    def _is_plausible_fixed_camera_detection(
        self,
        det: CameraDetection,
        predicted_xy: np.ndarray | None,
    ) -> bool:
        """Reject monocular CCTV projections that land off-map or far from the track."""
        if det.position_world is None:
            return False
        x = float(det.position_world[0].item())
        y = float(det.position_world[1].item())
        if not is_point_walkable(x, y, clearance=0.0):
            return False
        if predicted_xy is None:
            return True
        pred = torch.tensor(predicted_xy, dtype=det.position_world.dtype, device=det.position_world.device)
        dist = torch.norm(det.position_world[:2] - pred).item()
        velocity_allowance = 0.45 * float(np.linalg.norm(self._x[2:]))
        gate = max(1.25, self.camera_gate_distance + velocity_allowance)
        return dist <= gate

    def _collect_lidar_measurements(
        self,
        lidar_detections: list[LidarDetection],
        predicted_xy: np.ndarray | None,
    ) -> tuple[list[torch.Tensor], list[float]]:
        positions: list[torch.Tensor] = []
        weights: list[float] = []
        for det in lidar_detections:
            if not det.detected or det.position_world is None:
                continue
            det_weight = det.confidence * self.lidar_weight
            det_weight *= self._prediction_gate(
                det.position_world[:2],
                predicted_xy,
                base_distance=self.lidar_gate_distance,
                confidence=det.confidence,
                occlusion_score=0.0,
            )
            if det_weight <= 1.0e-4:
                continue
            positions.append(det.position_world)
            weights.append(det_weight)
        return positions, weights

    def _consolidate_measurements(
        self,
        positions: list[torch.Tensor],
        weights: list[float],
        predicted_xy: np.ndarray | None,
        radius: float,
    ) -> tuple[torch.Tensor | None, int, float]:
        """Collapse a modality's raw detections into a single consensus measurement."""
        if not positions:
            return None, 0, 0.0

        if len(positions) == 1:
            return positions[0], 1, weights[0]

        pos_t = torch.stack(positions)
        w_t = torch.tensor(weights, dtype=torch.float32, device=pos_t.device)

        if predicted_xy is not None:
            anchor = torch.tensor(predicted_xy, dtype=pos_t.dtype, device=pos_t.device)
            gate_radius = max(0.8, radius + 0.5)
            dists = torch.norm(pos_t[:, :2] - anchor.unsqueeze(0), dim=-1)
            mask = dists < gate_radius
            if mask.sum() >= 1:
                pos_t = pos_t[mask]
                w_t = w_t[mask]

        if pos_t.shape[0] > 1:
            median = pos_t[:, :2].median(dim=0).values
            dists = torch.norm(pos_t[:, :2] - median.unsqueeze(0), dim=-1)
            mask = dists < max(1.0, radius)
            if mask.sum() >= 1:
                pos_t = pos_t[mask]
                w_t = w_t[mask]

        w_norm = w_t / w_t.sum()
        consensus = (pos_t * w_norm.unsqueeze(-1)).sum(dim=0)
        return consensus, int(pos_t.shape[0]), float(w_t.sum().item())

    def _resolve_modality_disagreement(
        self,
        camera_consensus: torch.Tensor | None,
        camera_support: int,
        camera_weight_sum: float,
        lidar_consensus: torch.Tensor,
        lidar_weight_sum: float,
        predicted_xy: np.ndarray | None,
    ) -> float:
        """Down-weight LiDAR when it disagrees with stronger camera consensus."""
        if camera_consensus is None:
            return lidar_weight_sum

        disagreement = torch.norm(camera_consensus[:2] - lidar_consensus[:2]).item()
        if disagreement <= 1.15:
            return lidar_weight_sum
        if camera_weight_sum > 0.12:
            return 0.08 * lidar_weight_sum

        if predicted_xy is not None:
            pred = torch.tensor(predicted_xy, dtype=camera_consensus.dtype, device=camera_consensus.device)
            cam_pred_dist = torch.norm(camera_consensus[:2] - pred).item()
            lidar_pred_dist = torch.norm(lidar_consensus[:2] - pred).item()
            if cam_pred_dist + 0.4 < lidar_pred_dist:
                return 0.15 * lidar_weight_sum
            if lidar_pred_dist + 0.4 < cam_pred_dist and camera_support < 2:
                return lidar_weight_sum

        if camera_support >= 2 or camera_weight_sum >= lidar_weight_sum:
            return 0.15 * lidar_weight_sum
        return 0.4 * lidar_weight_sum

    # ------------------------------------------------------------------
    # Other helpers
    # ------------------------------------------------------------------

    def _prediction_gate(
        self,
        measurement_xy: torch.Tensor,
        predicted_xy: np.ndarray | None,
        base_distance: float,
        confidence: float,
        occlusion_score: float,
    ) -> float:
        """Soft-gate measurements against the predicted track state."""
        if predicted_xy is None:
            return 1.0

        pred = torch.tensor(predicted_xy, dtype=measurement_xy.dtype, device=measurement_xy.device)
        dist = torch.norm(measurement_xy - pred).item()
        velocity_allowance = 0.6 * float(np.linalg.norm(self._x[2:]))
        confidence_bonus = 0.4 * float(confidence)
        effective_gate = max(0.75, base_distance + velocity_allowance + confidence_bonus)
        effective_gate *= max(0.55, 1.0 - 0.45 * float(occlusion_score))

        if dist <= effective_gate:
            return 1.0
        if dist >= effective_gate * 2.0:
            return 0.0

        alpha = (dist - effective_gate) / max(effective_gate, 1.0e-6)
        return float((1.0 - alpha) ** 2)

    @staticmethod
    def _compute_error(
        estimated: Optional[torch.Tensor],
        ground_truth: Optional[torch.Tensor],
    ) -> float:
        """Compute 2D (XY plane) position error for tracking.

        Uses only X and Y components since Z-axis error from monocular
        depth estimation is not meaningful for floor-plane tracking.
        """
        if estimated is None or ground_truth is None:
            return float("inf")
        # Ensure same device
        gt = ground_truth.to(estimated.device) if ground_truth.device != estimated.device else ground_truth
        # 2D horizontal error (XY only)
        return torch.norm(estimated[:2] - gt[:2]).item()

    def _add_to_history(self, result: FusionResult):
        """Add result to history, maintaining max size."""
        self._history.append(result)
        if len(self._history) > self.history_size:
            self._history.pop(0)
