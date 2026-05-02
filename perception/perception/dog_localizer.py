"""Dog self-localization from onboard IMU propagation and LiDAR map matching."""

from __future__ import annotations

import math

import torch

from .scan_matching import (
    StaticMap2D,
    filter_local_scan_for_matching,
    match_scan,
    world_hits_to_local_points,
    wrap_angle,
)


def _clamp_cosine(value: float, eps: float = 1.0e-3) -> float:
    if abs(value) >= eps:
        return value
    return eps if value >= 0.0 else -eps


def _quat_from_euler(roll: float, pitch: float, yaw: float, device: torch.device) -> torch.Tensor:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    quat = torch.tensor(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=torch.float32,
        device=device,
    )
    return quat / torch.linalg.norm(quat)


def _euler_from_quat(quat_wxyz: torch.Tensor) -> tuple[float, float, float]:
    quat = quat_wxyz / torch.linalg.norm(quat_wxyz)
    w, x, y, z = [float(v) for v in quat]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, wrap_angle(yaw)


def _quat_rotate(quat_wxyz: torch.Tensor, vec_b: torch.Tensor) -> torch.Tensor:
    quat = quat_wxyz / torch.linalg.norm(quat_wxyz)
    w, x, y, z = quat
    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rot = torch.tensor(
        [
            [ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=torch.float32,
        device=vec_b.device,
    )
    return rot @ vec_b


def _yaw_from_quat(quat_wxyz: torch.Tensor) -> float:
    return _euler_from_quat(quat_wxyz)[2]


def _rotate_xy(vec_xy: torch.Tensor, yaw: float) -> torch.Tensor:
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = torch.tensor([[c, -s], [s, c]], dtype=torch.float32, device=vec_xy.device)
    return rot @ vec_xy


class DogLocalizer:
    """Estimate dog global planar pose, heading, and planar velocity from IMU and LiDAR."""

    def __init__(
        self,
        dog_name: str,
        dt: float,
        initial_pose: tuple[torch.Tensor, torch.Tensor] | None = None,
        lidar_mount_height: float = 0.35,
        static_map: StaticMap2D | None = None,
        lidar_update_steps: int = 6,
        imu_gravity_gain: float = 0.08,
        zupt_speed_threshold: float = 0.08,
        max_planar_speed_mps: float = 1.60,
        min_stable_match_score: float = 0.62,
        max_lidar_speed_mps: float = 1.80,
        max_lidar_yaw_rate_dps: float = 45.0,
        max_confident_lidar_yaw_rate_dps: float = 120.0,
        soft_yaw_hint_max_delta_deg: float = 14.0,
    ):
        self.dog_name = dog_name
        self.dt = float(dt)
        self.lidar_mount_height = float(lidar_mount_height)
        self.static_map = static_map
        self.lidar_update_steps = max(1, int(lidar_update_steps))
        self.imu_gravity_gain = float(imu_gravity_gain)
        self.zupt_speed_threshold = float(zupt_speed_threshold)
        self.max_planar_speed_mps = float(max_planar_speed_mps)
        self.min_stable_match_score = float(min_stable_match_score)
        self.max_lidar_speed_mps = float(max_lidar_speed_mps)
        self.max_lidar_yaw_rate = math.radians(float(max_lidar_yaw_rate_dps))
        self.max_confident_lidar_yaw_rate = math.radians(float(max_confident_lidar_yaw_rate_dps))
        self.soft_yaw_hint_max_delta = math.radians(float(soft_yaw_hint_max_delta_deg))

        self.pos_est = torch.zeros(3, dtype=torch.float32)
        self.vel_est = torch.zeros(3, dtype=torch.float32)
        self.quat_est = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

        self.initialized = False
        self.localized = False
        self.initial_z = 0.0
        self.roll_est = 0.0
        self.pitch_est = 0.0
        self.yaw_est = 0.0
        self.gyro_bias_z = 0.0

        self._step_count = 0
        self._last_lidar_step: int | None = None
        self._last_lidar_xy: torch.Tensor | None = None
        self._last_lidar_yaw: float | None = None
        self._last_match_score = 0.0
        self._last_match_inliers = 0
        self._accepted_score_ema = 0.0
        self._lidar_offset_body_xy = torch.zeros(2, dtype=torch.float32)
        self._lidar_yaw_offset = 0.0
        self._lidar_extrinsics_ready = False
        self._missed_lidar_updates = 0
        # Drift-free yaw reference integrated from the odometry hint. Used to
        # veto low-confidence LiDAR matches that try to twist yaw_est far from
        # the commanded heading at corner moments where ICP locally aliases.
        self._odom_yaw_ref: float | None = None

        if initial_pose is not None:
            self.reset(*initial_pose)

    def reset(self, initial_pos: torch.Tensor, initial_quat: torch.Tensor):
        """Reset the filter state from a known initial global pose."""
        initial_pos = initial_pos.detach().clone().to(dtype=torch.float32)
        initial_quat = initial_quat.detach().clone().to(dtype=torch.float32, device=initial_pos.device)

        self.pos_est = initial_pos.clone()
        self.vel_est = torch.zeros(3, dtype=torch.float32, device=initial_pos.device)
        self.quat_est = initial_quat / torch.linalg.norm(initial_quat)
        self.roll_est, self.pitch_est, self.yaw_est = _euler_from_quat(self.quat_est)
        self.initial_z = float(initial_pos[2].item())

        self.initialized = True
        self.localized = False
        self.gyro_bias_z = 0.0
        self._step_count = 0
        self._last_lidar_step = None
        self._last_lidar_xy = None
        self._last_lidar_yaw = None
        self._last_match_score = 0.0
        self._last_match_inliers = 0
        self._accepted_score_ema = 0.0
        self._lidar_offset_body_xy = torch.zeros(2, dtype=torch.float32, device=initial_pos.device)
        self._lidar_yaw_offset = 0.0
        self._lidar_extrinsics_ready = False
        self._missed_lidar_updates = 0
        self._odom_yaw_ref = float(self.yaw_est)

    def update(self, imu_data: dict, lidar_data: dict | None = None) -> dict:
        """Propagate with IMU and optionally correct using LiDAR scan matching."""
        if not self.initialized:
            return {}

        self._step_count += 1
        self._propagate_imu(imu_data)

        match_result = None
        if lidar_data is not None and self.static_map is not None:
            match_result = self._correct_with_lidar(lidar_data)

        return self._format_state(match_result)

    def get_estimate(self) -> dict:
        """Return the latest state estimate."""
        return self._format_state(match_result=None)

    def _propagate_imu(self, imu_data: dict):
        device = self.pos_est.device
        ang_vel_b = imu_data["ang_vel_b"].detach().to(device=device, dtype=torch.float32)
        lin_acc_b = imu_data["lin_acc_b"].detach().to(device=device, dtype=torch.float32)
        projected_gravity_b = imu_data["projected_gravity_b"].detach().to(device=device, dtype=torch.float32)
        odom_vel_w = imu_data.get("odom_vel_w")
        odom_ang_vel_w = imu_data.get("odom_ang_vel_w")
        if odom_vel_w is not None:
            odom_vel_w = odom_vel_w.detach().to(device=device, dtype=torch.float32)
        if odom_ang_vel_w is not None:
            odom_ang_vel_w = odom_ang_vel_w.detach().to(device=device, dtype=torch.float32)

        p = float(ang_vel_b[0].item())
        q = float(ang_vel_b[1].item())
        r = float(ang_vel_b[2].item())

        cos_pitch = _clamp_cosine(math.cos(self.pitch_est))
        roll_pred = self.roll_est + self.dt * (
            p + q * math.sin(self.roll_est) * math.tan(self.pitch_est) + r * math.cos(self.roll_est) * math.tan(self.pitch_est)
        )
        pitch_pred = self.pitch_est + self.dt * (q * math.cos(self.roll_est) - r * math.sin(self.roll_est))
        yaw_rate = (q * math.sin(self.roll_est) + r * math.cos(self.roll_est)) / cos_pitch
        yaw_rate_used = yaw_rate - self.gyro_bias_z
        if odom_ang_vel_w is not None and torch.isfinite(odom_ang_vel_w).all():
            odom_yaw_rate = float(odom_ang_vel_w[2].item())
            yaw_rate_used = 0.35 * yaw_rate_used + 0.65 * odom_yaw_rate
            if self._odom_yaw_ref is None:
                self._odom_yaw_ref = float(self.yaw_est)
            self._odom_yaw_ref = wrap_angle(self._odom_yaw_ref + self.dt * odom_yaw_rate)
        yaw_pred = wrap_angle(self.yaw_est + self.dt * yaw_rate_used)

        gravity_b = projected_gravity_b / max(float(torch.linalg.norm(projected_gravity_b).item()), 1.0e-6)
        gx, gy, gz = [float(v) for v in gravity_b]
        roll_gravity = math.atan2(gy, -gz)
        pitch_gravity = math.atan2(-gx, math.sqrt(gy * gy + gz * gz) + 1.0e-6)

        alpha = max(0.0, min(1.0, self.imu_gravity_gain))
        self.roll_est = wrap_angle((1.0 - alpha) * roll_pred + alpha * roll_gravity)
        self.pitch_est = (1.0 - alpha) * pitch_pred + alpha * pitch_gravity
        # Slow pull toward the drift-free odom yaw reference. Stops yaw_est
        # from accumulating IMU-bias drift during long LiDAR drought windows
        # (the kind that turned dog2's 70-step ICP-aliasing window into a 50°
        # yaw error). The pull is weak per step (5%) so an accepted LiDAR
        # match can still fully override yaw_est when ICP is reliable.
        if self._odom_yaw_ref is not None:
            yaw_pull_alpha = 0.05
            yaw_residual = wrap_angle(self._odom_yaw_ref - yaw_pred)
            yaw_pred = wrap_angle(yaw_pred + yaw_pull_alpha * yaw_residual)
        self.yaw_est = yaw_pred
        self.quat_est = _quat_from_euler(self.roll_est, self.pitch_est, self.yaw_est, device=device)

        acc_world = _quat_rotate(self.quat_est, lin_acc_b)
        acc_world = acc_world - torch.tensor([0.0, 0.0, 9.81], dtype=torch.float32, device=device)
        acc_world[2] = 0.0

        if torch.linalg.norm(acc_world[:2]) < 0.03:
            acc_world[:2] = 0.0

        self.vel_est[:2] += acc_world[:2] * self.dt
        if odom_vel_w is not None and torch.isfinite(odom_vel_w).all():
            self.vel_est[:2] = 0.15 * self.vel_est[:2] + 0.85 * odom_vel_w[:2]
        if self._is_stationary(ang_vel_b, lin_acc_b):
            self.vel_est[:2] *= 0.25
            if torch.linalg.norm(self.vel_est[:2]) < self.zupt_speed_threshold:
                self.vel_est[:2] = 0.0

        # Keep the dead-reckoned velocity bounded during LiDAR dropouts.
        lidar_recent = self._last_lidar_step is not None and (self._step_count - self._last_lidar_step) <= self.lidar_update_steps
        self.vel_est[:2] *= 0.98 if lidar_recent else 0.90
        self._clamp_planar_velocity()

        self.pos_est[:2] += self.vel_est[:2] * self.dt
        self.pos_est[2] = self.initial_z
        self.vel_est[2] = 0.0

    def _is_stationary(self, ang_vel_b: torch.Tensor, lin_acc_b: torch.Tensor) -> bool:
        planar_specific_force = float(torch.linalg.norm(lin_acc_b[:2]).item())
        yaw_rate = abs(float(ang_vel_b[2].item()))
        planar_speed = float(torch.linalg.norm(self.vel_est[:2]).item())
        return yaw_rate < 0.05 and planar_specific_force < 0.25 and planar_speed < self.zupt_speed_threshold * 1.5

    def _correct_with_lidar(self, lidar_data: dict):
        device = self.pos_est.device
        sensor_pos_w = lidar_data["pos_w"].detach().to(device=device, dtype=torch.float32)
        sensor_quat_w = lidar_data["quat_w"].detach().to(device=device, dtype=torch.float32)
        if sensor_pos_w.ndim > 1:
            sensor_pos_w = sensor_pos_w[0]
        if sensor_quat_w.ndim > 1:
            sensor_quat_w = sensor_quat_w[0]

        if not self._lidar_extrinsics_ready:
            sensor_yaw = _yaw_from_quat(sensor_quat_w)
            self._lidar_yaw_offset = wrap_angle(sensor_yaw - self.yaw_est)
            offset_world_xy = sensor_pos_w[:2] - self.pos_est[:2]
            self._lidar_offset_body_xy = _rotate_xy(offset_world_xy, -self.yaw_est)
            self._lidar_extrinsics_ready = True

        local_points = world_hits_to_local_points(
            lidar_data["hit_points"],
            sensor_pos_w,
            sensor_quat_w,
        )
        scan_xy = filter_local_scan_for_matching(local_points)

        predicted_sensor_xy = self.pos_est[:2] + _rotate_xy(self._lidar_offset_body_xy, self.yaw_est)
        predicted_sensor_pose = (
            float(predicted_sensor_xy[0].item()),
            float(predicted_sensor_xy[1].item()),
            wrap_angle(self.yaw_est + self._lidar_yaw_offset),
        )
        primary_match = match_scan(predicted_sensor_pose, scan_xy, self.static_map)
        match_result = primary_match

        primary_bootstrap_accept = (
            not primary_match.accepted
            and not self.localized
            and primary_match.inliers >= 120
            and math.isfinite(primary_match.rmse)
            and primary_match.rmse < 1.10
            and primary_match.score >= 0.18
        )
        if primary_bootstrap_accept:
            match_result = type(primary_match)(
                pose_xyyaw=primary_match.pose_xyyaw,
                score=primary_match.score,
                inliers=primary_match.inliers,
                rmse=primary_match.rmse,
                accepted=True,
                xy_correction=primary_match.xy_correction,
                yaw_correction=primary_match.yaw_correction,
                used_prediction_only=False,
            )
        elif not primary_match.accepted and not self.localized and (
            primary_match.inliers < 80 or primary_match.score < 0.12
        ):
            candidate_poses = [predicted_sensor_pose]
            for yaw_offset in (
                math.pi / 2.0,
                -math.pi / 2.0,
                math.pi,
                math.pi / 4.0,
                -math.pi / 4.0,
            ):
                candidate_poses.append(
                    (
                        predicted_sensor_pose[0],
                        predicted_sensor_pose[1],
                        wrap_angle(predicted_sensor_pose[2] + yaw_offset),
                    )
                )

            broad_candidates = [
                match_scan(
                    candidate_pose,
                    scan_xy,
                    self.static_map,
                    max_iterations=8,
                    max_correspondence_dist=1.75,
                    max_translation_delta=2.5,
                    max_yaw_delta=math.pi,
                )
                for candidate_pose in candidate_poses
            ]
            best_broad_match = max(
                broad_candidates,
                key=lambda result: (
                    int(result.accepted),
                    result.score,
                    result.inliers,
                    -result.rmse if math.isfinite(result.rmse) else float("-inf"),
                ),
            )
            broad_bootstrap_accept = (
                not best_broad_match.accepted
                and best_broad_match.inliers >= 120
                and math.isfinite(best_broad_match.rmse)
                and best_broad_match.rmse < 1.10
                and best_broad_match.score >= max(primary_match.score + 0.08, 0.22)
            )
            if broad_bootstrap_accept:
                best_broad_match = type(best_broad_match)(
                    pose_xyyaw=best_broad_match.pose_xyyaw,
                    score=best_broad_match.score,
                    inliers=best_broad_match.inliers,
                    rmse=best_broad_match.rmse,
                    accepted=True,
                    xy_correction=best_broad_match.xy_correction,
                    yaw_correction=best_broad_match.yaw_correction,
                    used_prediction_only=False,
                )
            if best_broad_match.accepted:
                match_result = best_broad_match

        self._last_match_score = match_result.score
        self._last_match_inliers = match_result.inliers

        if not match_result.accepted:
            self._missed_lidar_updates += 1
            if self._missed_lidar_updates >= 3:
                self.localized = False
            self._maybe_apply_soft_yaw_hint(match_result)
            return match_result

        corrected_sensor_xy = torch.tensor(match_result.pose_xyyaw[:2], dtype=torch.float32, device=device)
        corrected_sensor_yaw = float(match_result.pose_xyyaw[2])
        corrected_yaw = wrap_angle(corrected_sensor_yaw - self._lidar_yaw_offset)
        corrected_xy = corrected_sensor_xy - _rotate_xy(self._lidar_offset_body_xy, corrected_yaw)
        if not self._is_stable_lidar_match(match_result, corrected_xy, corrected_yaw):
            self._missed_lidar_updates += 1
            if self._missed_lidar_updates >= 3:
                self.localized = False
            rejected_match = type(match_result)(
                pose_xyyaw=match_result.pose_xyyaw,
                score=match_result.score,
                inliers=match_result.inliers,
                rmse=match_result.rmse,
                accepted=False,
                xy_correction=match_result.xy_correction,
                yaw_correction=match_result.yaw_correction,
                used_prediction_only=True,
            )
            self._maybe_apply_soft_yaw_hint(rejected_match)
            return rejected_match

        if self._last_lidar_xy is not None and self._last_lidar_step is not None:
            elapsed = max((self._step_count - self._last_lidar_step) * self.dt, self.dt)
            measured_vel_xy = (corrected_xy - self._last_lidar_xy) / elapsed
            measured_speed = float(torch.linalg.norm(measured_vel_xy).item())
            if measured_speed > self.max_lidar_speed_mps:
                measured_vel_xy *= self.max_lidar_speed_mps / max(measured_speed, 1.0e-6)

            # The displacement carries both real motion and the position
            # correction. When the correction dominates the displacement, the
            # implied velocity is mostly noise, so reduce its weight.
            correction_norm = math.hypot(*match_result.xy_correction)
            correction_speed = correction_norm / elapsed
            quality = max(0.0, min(1.0, (match_result.score - 0.40) / 0.40))
            disagreement = float(
                torch.linalg.norm(measured_vel_xy - self.vel_est[:2]).item()
            )

            weight = 0.10 + 0.30 * quality
            if correction_speed > 0.40:
                weight *= max(0.20, 1.0 - (correction_speed - 0.40) / 0.80)
            if elapsed > 4.0 * self.lidar_update_steps * self.dt:
                weight *= 0.30
            if disagreement > 0.80:
                weight *= 0.30
            weight = float(np_clip(weight, 0.0, 0.45))
            self.vel_est[:2] = (1.0 - weight) * self.vel_est[:2] + weight * measured_vel_xy
        else:
            self.vel_est[:2] *= 0.5
        self._clamp_planar_velocity()

        self.pos_est[:2] = corrected_xy
        self.pos_est[2] = self.initial_z
        self.vel_est[2] = 0.0

        yaw_delta = wrap_angle(corrected_yaw - self.yaw_est)
        self.yaw_est = corrected_yaw
        bias_update = yaw_delta / max(self.lidar_update_steps * self.dt, self.dt)
        self.gyro_bias_z = float(np_clip(self.gyro_bias_z - 0.05 * bias_update, -0.5, 0.5))
        self.quat_est = _quat_from_euler(self.roll_est, self.pitch_est, self.yaw_est, device=device)

        self.localized = True
        self._missed_lidar_updates = 0
        self._last_lidar_xy = corrected_xy.clone()
        self._last_lidar_yaw = corrected_yaw
        self._last_lidar_step = self._step_count
        self._accepted_score_ema = match_result.score if self._accepted_score_ema <= 0.0 else (0.85 * self._accepted_score_ema + 0.15 * match_result.score)
        return match_result

    def _clamp_planar_velocity(self):
        planar_speed = float(torch.linalg.norm(self.vel_est[:2]).item())
        if planar_speed > self.max_planar_speed_mps:
            self.vel_est[:2] *= self.max_planar_speed_mps / max(planar_speed, 1.0e-6)

    def _maybe_apply_soft_yaw_hint(self, match_result) -> bool:
        if not self.localized:
            return False
        if match_result.score < max(self.min_stable_match_score, 0.72):
            return False
        if match_result.inliers < 120:
            return False
        hinted_yaw = wrap_angle(float(match_result.pose_xyyaw[2]) - self._lidar_yaw_offset)
        yaw_delta = wrap_angle(hinted_yaw - self.yaw_est)
        if abs(yaw_delta) < math.radians(2.0) or abs(yaw_delta) > math.radians(18.0):
            return False

        # Same odom-yaw guard as in `_is_stable_lidar_match`: don't twist
        # heading toward a hint that disagrees with the commanded heading.
        if self._odom_yaw_ref is not None:
            odom_disagreement = abs(wrap_angle(hinted_yaw - self._odom_yaw_ref))
            if odom_disagreement > math.radians(15.0):
                return False

        yaw_step = np_clip(yaw_delta, -self.soft_yaw_hint_max_delta, self.soft_yaw_hint_max_delta)
        self.yaw_est = wrap_angle(self.yaw_est + yaw_step)
        self.quat_est = _quat_from_euler(self.roll_est, self.pitch_est, self.yaw_est, device=self.pos_est.device)
        bias_update = yaw_step / max(self.lidar_update_steps * self.dt, self.dt)
        self.gyro_bias_z = float(np_clip(self.gyro_bias_z - 0.02 * bias_update, -0.5, 0.5))
        return True

    def _is_stable_lidar_match(self, match_result, corrected_xy: torch.Tensor, corrected_yaw: float) -> bool:
        if not self.localized:
            return True

        correction_xy_norm = math.hypot(*match_result.xy_correction)
        yaw_correction_abs = abs(float(match_result.yaw_correction))
        small_correction = correction_xy_norm <= 0.08 and yaw_correction_abs <= math.radians(4.0)
        high_confidence_match = (
            match_result.score >= max(self.min_stable_match_score, 0.75)
            and match_result.inliers >= 120
            and correction_xy_norm <= 0.30
        )
        score_floor = self.min_stable_match_score
        if self._accepted_score_ema > 0.0:
            score_floor = max(score_floor, 0.78 * self._accepted_score_ema)

        if match_result.score < score_floor and not small_correction:
            return False

        # Reject yaw-twisting matches that disagree with the drift-free odom
        # yaw integral when the match is only mediocre. This stops corner-time
        # ICP local minima from rotating us 30-50 degrees off truth.
        if self._odom_yaw_ref is not None and not high_confidence_match:
            odom_disagreement = abs(wrap_angle(corrected_yaw - self._odom_yaw_ref))
            allowed = math.radians(15.0)
            if match_result.score < 0.85:
                allowed = math.radians(12.0)
            if odom_disagreement > allowed and not small_correction:
                return False

        if self._last_lidar_xy is not None and self._last_lidar_step is not None:
            elapsed = max((self._step_count - self._last_lidar_step) * self.dt, self.dt)
            measured_speed = float(torch.linalg.norm(corrected_xy - self._last_lidar_xy).item()) / elapsed
            if measured_speed > self.max_lidar_speed_mps and not small_correction:
                return False

        if self._last_lidar_yaw is not None:
            if self._last_lidar_step is None:
                elapsed = self.dt
            else:
                elapsed = max((self._step_count - self._last_lidar_step) * self.dt, self.dt)
            measured_yaw_rate = abs(wrap_angle(corrected_yaw - self._last_lidar_yaw)) / elapsed
            allowed_yaw_rate = self.max_confident_lidar_yaw_rate if high_confidence_match else self.max_lidar_yaw_rate
            if measured_yaw_rate > allowed_yaw_rate and not small_correction:
                return False

        return True

    def _format_state(self, match_result) -> dict:
        """Return the latest dog state.

        `pos`, `vel`, and `quat` are in the global world frame (matching
        Isaac Lab's `root_pos_w`, `root_lin_vel_w`, `root_quat_w`). `euler`
        is the world-frame (roll, pitch, yaw) triple in radians. Z position
        is held at the initial root height (`z_locked=True`) because the
        localizer is planar.
        """
        score = match_result.score if match_result is not None else 0.0
        inliers = match_result.inliers if match_result is not None else 0
        yaw_correction = match_result.yaw_correction if match_result is not None else 0.0
        lidar_corrected = bool(match_result.accepted) if match_result is not None else False

        return {
            "pos": self.pos_est.clone(),
            "quat": self.quat_est.clone(),
            "vel": self.vel_est.clone(),
            "speed": torch.linalg.norm(self.vel_est).item(),
            "euler": (self.roll_est, self.pitch_est, self.yaw_est),
            "localized": self.localized,
            "detected": self.localized,
            "lidar_corrected": lidar_corrected,
            "scan_match_score": score,
            "scan_inliers": inliers,
            "yaw_correction": yaw_correction,
            "z_locked": True,
            "only_imu_prediction": not lidar_corrected,
        }


def np_clip(value: float, lower: float, upper: float) -> float:
    """Scalar clip helper without importing numpy at module scope."""
    return min(max(value, lower), upper)
