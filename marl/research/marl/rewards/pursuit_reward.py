"""
pursuit_reward.py
Reward functions for the pursuit task.

Three classes:
  - PursuitReward      : original team-shared scalar (kept for backward compat)
  - RolePursuitReward  : per-role per-agent reward (PURSUER vs ENCIRCLER)
  - IntruderReward     : reward for the intruder policy (negative of pursuit progress)

The PURSUER role focuses on closing the distance to the target.
The ENCIRCLER role focuses on geometric encirclement: staying on the opposite side
of the target from the pursuer, and blocking the target's escape direction.
Both still get a shared team capture/time bonus when the dual-90° encirclement
condition is satisfied.
"""
from __future__ import annotations

import numpy as np


PURSUER = 0
ENCIRCLER = 1


# ============================================================================
# Original (kept for backward compatibility with the 21D / shared-reward setup)
# ============================================================================
class PursuitReward:
    def __init__(self, cfg: dict):
        self.w_dist      = float(cfg.get("w_distance",           0.5))
        self.w_capture   = float(cfg.get("w_capture",            200.0))
        self.w_step      = float(cfg.get("w_step",               -0.02))
        self.w_proximity = float(cfg.get("w_proximity",          -3.0))
        self.sep_thresh  = float(cfg.get("separation_threshold",  2.5))
        self.w_pin       = float(cfg.get("w_pin",                2.0))
        self.w_time      = float(cfg.get("w_time",               100.0))

    def compute(
        self,
        agent_pos:    np.ndarray,
        target_pos:   np.ndarray,
        capture_radius: float = 1.5,
        dists_before: np.ndarray = None,
        dists_after:  np.ndarray = None,
        step_count:   int = 0,
        max_steps:    int = 500,
    ) -> np.ndarray:
        n = len(agent_pos)
        if dists_after is None:
            dists_after = np.linalg.norm(agent_pos - target_pos, axis=1)

        if dists_before is not None:
            progress   = float(np.mean(dists_before - dists_after))
            r_progress = self.w_dist * progress * 10.0
        else:
            r_progress = self.w_dist * (-float(np.mean(dists_after)))

        r_pin = 0.0
        for d in dists_after:
            if d <= 2.0:
                r_pin += self.w_pin

        r_capture = 0.0
        r_time_bonus = 0.0
        if n >= 2:
            d1 = float(dists_after[0]); d2 = float(dists_after[1])
            if d1 <= capture_radius and d2 <= capture_radius:
                v1 = agent_pos[0] - target_pos
                v2 = agent_pos[1] - target_pos
                if d1 > 0.1 and d2 > 0.1:
                    cos_theta = np.dot(v1, v2) / (d1 * d2)
                    if cos_theta <= 0.0:
                        angle_quality = (1.0 - cos_theta) / 2.0
                        r_capture = self.w_capture * float(angle_quality)
                        time_fraction = max(0.0, (max_steps - step_count) / max_steps)
                        r_time_bonus = self.w_time * time_fraction
        else:
            if np.any(dists_after <= capture_radius):
                r_capture = self.w_capture
                time_fraction = max(0.0, (max_steps - step_count) / max_steps)
                r_time_bonus = self.w_time * time_fraction

        r_proximity = 0.0
        if n >= 2:
            d_agents = float(np.linalg.norm(agent_pos[0] - agent_pos[1]))
            if d_agents < self.sep_thresh:
                r_proximity = self.w_proximity * (self.sep_thresh - d_agents)

        r_step = self.w_step
        total = r_progress + r_pin + r_capture + r_time_bonus + r_proximity + r_step
        return np.full(n, total, dtype=np.float32)


# ============================================================================
# NEW: per-role reward
# ============================================================================
class RolePursuitReward:
    """Per-agent reward where the formula depends on role (PURSUER vs ENCIRCLER).

    PURSUER reward:
        + w_dist_pursuer * Δd_self           # strong dense progress
        + w_pin_pursuer  * 1[d_self <= pin_threshold]
        + w_lag_pursuer  * max(0, d_self - lag_threshold)   # penalty if dragging
        + (shared team terms)

    ENCIRCLER reward:
        + w_angle_encircler  * (1 - cos θ)/2   # θ = angle(pursuer→target, encircler→target)
        + w_cutoff_encircler * cutoff_score    # how well it blocks target's escape direction
        + w_dist_encircler   * Δd_self * 0.3   # weak progress (still want to be in the area)
        + (shared team terms)

    Shared team terms (added to BOTH roles):
        + capture_bonus      (when dual-90° encirclement satisfied)
        + time_bonus         (decays with step count)
        + proximity_penalty  (when agents overlap < sep_threshold)
        + step_penalty
    """

    def __init__(self, cfg: dict):
        # Pursuer
        self.w_dist_p   = float(cfg.get("w_dist_pursuer",   1.5))
        self.w_pin_p    = float(cfg.get("w_pin_pursuer",    1.0))   # halved; pin no longer dominates
        self.w_lag_p    = float(cfg.get("w_lag_pursuer",   -0.5))
        self.pin_thresh = float(cfg.get("pin_threshold",    1.5))   # align with capture_radius — no hover trap
        self.lag_thresh = float(cfg.get("lag_threshold",    5.0))

        # Encircler
        self.w_angle_e  = float(cfg.get("w_angle_encircler",   2.0))   # was 3.0; less pure-hover incentive
        self.w_cutoff_e = float(cfg.get("w_cutoff_encircler",  0.0))   # DROPPED — coupled to noisy target_vel
        self.w_dist_e   = float(cfg.get("w_dist_encircler",    0.8))   # was 0.3; encircler must close in too
        self.w_commit_e = float(cfg.get("w_commit_encircler",  5.0))   # NEW: strong reward when at capture-ready position
        self.commit_radius_mul = float(cfg.get("commit_radius_mul", 1.5))  # commit zone = capture_radius × this

        # Shared team
        self.w_capture   = float(cfg.get("w_capture",     400.0))   # 2x — capture must dominate dense
        self.w_time      = float(cfg.get("w_time",        200.0))   # 2x
        self.w_proximity = float(cfg.get("w_proximity",    -0.5))
        self.sep_thresh  = float(cfg.get("separation_threshold", 1.5))
        self.w_step      = float(cfg.get("w_step",         -0.1))    # restore higher step penalty
        # Co-presence bonus: when both agents within capture_radius, regardless of angle.
        # This rewards "committing to capture" so they don't hover at the edge.
        self.w_copresence = float(cfg.get("w_copresence",  3.0))
        # Anti-stuck: penalty applied when both agents have been inside capture_radius
        # for stuck_threshold consecutive steps without satisfying the 90° rule.
        # Forces the policy out of the "almost-capture but never commits" local optimum.
        self.w_stuck         = float(cfg.get("w_stuck",         -10.0))
        self.stuck_threshold = int(cfg.get("stuck_threshold",     40))
        # Time pressure: step penalty escalates linearly with step_count.
        self.time_escalation = float(cfg.get("time_escalation_rate", 0.002))
        # Same-side soft penalty: when both agents are in striking range AND
        # geometrically on the same side of the target (cos > threshold),
        # apply a per-step penalty that scales with how stacked they are.
        # Targets the failure mode where agents arrive from the same direction
        # and never split to encircle (e.g. after chasing intruder into a wall).
        self.w_same_side       = float(cfg.get("w_same_side",       -1.5))
        self.same_side_thresh  = float(cfg.get("same_side_thresh",   0.5))
        self.same_side_radius  = float(cfg.get("same_side_radius",   4.0))
        # NEW (v12): Facing reward — rewards each agent for orienting heading
        # toward the intruder when within engagement range. Required for the
        # capture rule which now demands "facing" cosine ≥ face_capture_thresh.
        self.w_face_pursuer   = float(cfg.get("w_face_pursuer",   0.8))
        self.w_face_encircler = float(cfg.get("w_face_encircler", 0.5))
        self.face_reward_radius = float(cfg.get("face_reward_radius", 5.0))
        # NEW (v12): Hold-progress reward — tiny per-step bonus while the
        # geometric capture conditions are satisfied. Encourages locking in
        # the position for the 2-second capture confirmation.
        self.w_hold_step      = float(cfg.get("w_hold_step",      3.0))

    def compute_per_agent(
        self,
        agent_pos:      np.ndarray,   # (n, 2)
        target_pos:     np.ndarray,   # (2,)
        target_vel:     np.ndarray,   # (2,)
        roles:          np.ndarray,   # (n,) int  PURSUER / ENCIRCLER
        capture_radius: float,
        dists_before:   np.ndarray,
        dists_after:    np.ndarray,
        step_count:     int,
        max_steps:      int,
        stuck_steps:    int = 0,      # consecutive both-in-capture-but-no-capture frames
        agent_yaw:      np.ndarray = None,   # (n,) heading rad — for facing reward
        capture_geom_now: bool = False,       # True when capture geometry holds this frame
    ) -> np.ndarray:
        n = len(agent_pos)
        rewards = np.zeros(n, dtype=np.float32)
        if n != 2:
            # Fallback: only role-aware logic implemented for n=2
            return rewards

        # Identify pursuer / encircler indices
        idx_p = int(np.where(roles == PURSUER)[0][0]) if PURSUER in roles else 0
        idx_e = int(np.where(roles == ENCIRCLER)[0][0]) if ENCIRCLER in roles else 1

        d_p_before, d_e_before = float(dists_before[idx_p]), float(dists_before[idx_e])
        d_p_after,  d_e_after  = float(dists_after[idx_p]),  float(dists_after[idx_e])

        # ---- Pursuer reward ----
        r_p = 0.0
        # Dense progress (own distance)
        r_p += self.w_dist_p * (d_p_before - d_p_after) * 10.0
        # Pin — gated by:
        #   (a) distance window 0.3m..pin_thresh  (no reward for stacking ON intruder)
        #   (b) encircler is on opposite side (cos θ between v_p and v_e ≤ 0.5, ~60°+)
        # This makes pin a *team-coordinated* reward, not a solo "stay close" reward.
        if 0.3 < d_p_after <= self.pin_thresh:
            v_p_t = agent_pos[idx_p] - target_pos
            v_e_t = agent_pos[idx_e] - target_pos
            np_p_t = float(np.linalg.norm(v_p_t))
            np_e_t = float(np.linalg.norm(v_e_t))
            if np_p_t > 0.1 and np_e_t > 0.1:
                cos_pe = float(np.dot(v_p_t, v_e_t) / (np_p_t * np_e_t))
                if cos_pe <= 0.5:
                    r_p += self.w_pin_p
        # Lag penalty
        if d_p_after > self.lag_thresh:
            r_p += self.w_lag_p * (d_p_after - self.lag_thresh)

        # ---- Encircler reward ----
        r_e = 0.0
        # 1) Angle-of-encirclement: θ between (pursuer→target) and (encircler→target)
        #    Smooth distance decay (no hard cutoff) so encircler still gets a small
        #    pull at 6m+ but a much steeper gradient near capture range.
        v_p = agent_pos[idx_p] - target_pos
        v_e = agent_pos[idx_e] - target_pos
        np_p, np_e = float(np.linalg.norm(v_p)), float(np.linalg.norm(v_e))
        if np_p > 0.1 and np_e > 0.1:
            cos_theta = float(np.dot(v_p, v_e) / (np_p * np_e))
            angle_quality = (1.0 - cos_theta) / 2.0   # ∈ [0, 1]
            # Smooth decay: 1.0 at 0m, 0.67 at 1m, 0.5 at 2m, 0.33 at 4m, 0.25 at 6m
            distance_factor = 1.0 / (1.0 + d_e_after * 0.5)
            r_e += self.w_angle_e * (angle_quality ** 2) * distance_factor

        # 2) Commit bonus — strong reward only when encircler is *near capture range*
        #    AND geometrically opposite. This breaks the "stay safely at distance"
        #    equilibrium where encircler hovers at 3m+ never committing.
        commit_radius = capture_radius * self.commit_radius_mul   # e.g. 1.5 × 1.5 = 2.25 m
        if d_e_after <= commit_radius and np_p > 0.1 and np_e > 0.1:
            cos_theta_c = float(np.dot(v_p, v_e) / (np_p * np_e))
            if cos_theta_c < 0.0:                  # only when angle > 90°
                commit_quality = -cos_theta_c       # 0 at 90°, 1 at 180°
                r_e += self.w_commit_e * commit_quality

        # 2) Cutoff: DROPPED — was coupled to noisy target_vel direction, caused
        #    encircler to oscillate whenever the intruder changed heading.
        if self.w_cutoff_e != 0.0:
            spd = float(np.linalg.norm(target_vel))
            if spd > 0.3 and np_e > 0.1:
                target_dir = target_vel / spd
                enc_dir    = (agent_pos[idx_e] - target_pos) / np_e
                cutoff_score = max(0.0, float(np.dot(enc_dir, target_dir)))
                r_e += self.w_cutoff_e * cutoff_score

        # 3) Strong progress for encircler — must close in for capture
        r_e += self.w_dist_e * (d_e_before - d_e_after) * 10.0

        # ---- Facing reward (v12) ----
        # Both agents are rewarded for orienting heading toward target when
        # within engagement range. Required because the new capture rule
        # requires cos(heading, target_dir) ≥ face_capture_thresh.
        if agent_yaw is not None and self.face_reward_radius > 0:
            for idx, w_face in ((idx_p, self.w_face_pursuer),
                                (idx_e, self.w_face_encircler)):
                to_t = target_pos - agent_pos[idx]
                d_t = float(np.linalg.norm(to_t))
                if 1e-3 < d_t <= self.face_reward_radius:
                    head = np.array([np.cos(float(agent_yaw[idx])),
                                     np.sin(float(agent_yaw[idx]))])
                    cos_face = float(np.dot(head, to_t / d_t))
                    contribution = w_face * max(0.0, cos_face)
                    if idx == idx_p:
                        r_p += contribution
                    else:
                        r_e += contribution

        # ---- Shared team terms ----
        # Co-presence: both within capture_radius. Quality is SQUARED so that the
        # ~80° "almost-but-not-quite" zone gives much less reward than true 180°
        # encirclement. This eliminates the local optimum where agents settle at
        # ~80° and never commit the final positioning step to trigger capture.
        #   q = (1 - cosθ) / 2  ∈ [0,1];  0 stacked, 0.5 at 90°, 1 at 180°
        #   reward = w_copresence × q²  → 0 stacked, 0.75 at 90°, 3.0 at 180°
        # Numerical guard: if pursuer is *on top of* the target (np_p < 0.1), no
        # copresence — there's no meaningful angle when v_p ≈ 0, plus we don't
        # want to reward "hover-on-target" exploit.
        r_copresence = 0.0
        if d_p_after <= capture_radius and d_e_after <= capture_radius:
            if np_p > 0.1 and np_e > 0.1:
                cos_theta_cp = float(np.dot(v_p, v_e) / (np_p * np_e))
                encirclement_quality = max(0.0, (1.0 - cos_theta_cp) / 2.0)
                r_copresence = self.w_copresence * (encirclement_quality ** 2)

        # Capture (still requires the 90° angle)
        r_capture = 0.0
        r_time_bonus = 0.0
        if d_p_after <= capture_radius and d_e_after <= capture_radius:
            if np_p > 0.1 and np_e > 0.1:
                cos_theta = float(np.dot(v_p, v_e) / (np_p * np_e))
                if cos_theta <= 0.0:
                    angle_quality = (1.0 - cos_theta) / 2.0
                    r_capture = self.w_capture * angle_quality
                    time_fraction = max(0.0, (max_steps - step_count) / max_steps)
                    r_time_bonus = self.w_time * time_fraction

        # Proximity penalty
        r_proximity = 0.0
        d_agents = float(np.linalg.norm(agent_pos[0] - agent_pos[1]))
        if d_agents < self.sep_thresh:
            r_proximity = self.w_proximity * (self.sep_thresh - d_agents)

        # Step penalty — escalates linearly with time pressure
        time_mult = 1.0 + self.time_escalation * float(step_count)   # at 500 ≈ 2.0
        r_step = self.w_step * time_mult

        # Same-side stacking penalty
        r_same_side = 0.0
        if (d_p_after <= self.same_side_radius
                and d_e_after <= self.same_side_radius
                and np_p > 0.1 and np_e > 0.1):
            cos_ps = float(np.dot(v_p, v_e) / (np_p * np_e))
            if cos_ps > self.same_side_thresh:
                # 0 at threshold, 1 at fully stacked
                stack_factor = (cos_ps - self.same_side_thresh) / (1.0 - self.same_side_thresh)
                stack_factor = max(0.0, min(1.0, stack_factor))
                r_same_side = self.w_same_side * stack_factor

        # Anti-stuck penalty — fires when stuck_steps >= threshold, escalates
        r_stuck = 0.0
        if stuck_steps >= self.stuck_threshold:
            # constant per-step beyond threshold; use stuck count to scale slightly
            over = stuck_steps - self.stuck_threshold + 1
            r_stuck = self.w_stuck * (1.0 + 0.05 * (over - 1))  # -10, -10.5, -11, ...

        # Hold-progress reward — small per-step bonus while capture geometry
        # is satisfied. This makes the 2-second lock-in profitable and gives
        # the policy a continuous gradient toward holding the position.
        r_hold = self.w_hold_step if capture_geom_now else 0.0

        team_shared = (r_copresence + r_capture + r_time_bonus
                       + r_proximity + r_step + r_stuck + r_same_side + r_hold)
        r_p += team_shared
        r_e += team_shared

        rewards[idx_p] = r_p
        rewards[idx_e] = r_e
        return rewards


# ============================================================================
# NEW: intruder reward (mirror of pursuit reward — survival good, capture bad)
# ============================================================================
class IntruderReward:
    """Reward for the intruder policy.

    Components:
        + w_alive    each step (survival bonus)
        + w_distance * Δ(min agent distance)   # increasing distance from nearest pursuer = good
        + w_captured (one-time on capture, big negative)
        + w_wall     when intruder hits wall and bounces (small negative)
    """

    def __init__(self, cfg: dict):
        self.w_alive     = float(cfg.get("w_intruder_alive",     0.05))
        self.w_distance  = float(cfg.get("w_intruder_distance",  0.5))
        self.w_captured  = float(cfg.get("w_intruder_captured", -100.0))
        self.w_wall      = float(cfg.get("w_intruder_wall",     -0.1))

    def compute(
        self,
        min_dist_before: float,
        min_dist_after:  float,
        captured:        bool,
        hit_wall:        bool,
    ) -> float:
        r = self.w_alive
        # Distance gain (positive when intruder moves AWAY from nearest pursuer)
        r += self.w_distance * (min_dist_after - min_dist_before) * 10.0
        if captured:
            r += self.w_captured
        if hit_wall:
            r += self.w_wall
        return float(r)
