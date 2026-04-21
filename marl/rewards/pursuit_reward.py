"""
pursuit_reward.py
Shared team reward for the pursuit task.
Both agents receive the same scalar reward per step.

Components:
  R = w_dist     * dense_progress_reward    (close the gap each step)
    + w_capture  * capture_bonus            (big reward on interception)
    + w_proximity* proximity_penalty        (push agents apart when overlapping)
    + w_step     * step_penalty             (encourage speed)
"""
from __future__ import annotations

import numpy as np


class PursuitReward:
    def __init__(self, cfg: dict):
        self.w_dist      = float(cfg.get("w_distance",           0.5))
        self.w_capture   = float(cfg.get("w_capture",            200.0))
        self.w_step      = float(cfg.get("w_step",               -0.02))
        # Proximity penalty: penalise when agents are too close to each other
        # (NOT a spread bonus — only penalises overlap, doesn't reward being far)
        self.w_proximity = float(cfg.get("w_proximity",          -3.0))
        self.sep_thresh  = float(cfg.get("separation_threshold",  2.5))   # metres

    def compute(
        self,
        agent_pos:    np.ndarray,          # (n_agents, 2)
        target_pos:   np.ndarray,          # (2,)
        capture_radius: float = 1.0,
        dists_before: np.ndarray = None,   # (n_agents,) distance before step
        dists_after:  np.ndarray = None,   # (n_agents,) distance after step
    ) -> np.ndarray:
        n = len(agent_pos)
        if dists_after is None:
            dists_after = np.linalg.norm(agent_pos - target_pos, axis=1)

        # 1. Dense progress: reward closing distance each step
        if dists_before is not None:
            progress   = float(np.mean(dists_before - dists_after))
            r_progress = self.w_dist * progress * 10.0
        else:
            r_progress = self.w_dist * (-float(np.mean(dists_after)))

        # 2. Capture bonus
        r_capture = self.w_capture if np.any(dists_after <= capture_radius) else 0.0

        # 3. Proximity penalty — only fires when agents overlap (< sep_thresh)
        #    Gradient pushes agents to maintain >= sep_thresh separation,
        #    but gives NO reward for being arbitrarily far apart (no hacking).
        r_proximity = 0.0
        if n >= 2:
            d_agents = float(np.linalg.norm(agent_pos[0] - agent_pos[1]))
            if d_agents < self.sep_thresh:
                r_proximity = self.w_proximity * (self.sep_thresh - d_agents)
                # e.g. at 0m apart   → -3.0 * 2.5 = -7.5
                # at 1m apart        → -3.0 * 1.5 = -4.5
                # at 2.5m+ apart     →  0

        # 4. Step penalty
        r_step = self.w_step

        total = r_progress + r_capture + r_proximity + r_step
        return np.full(n, total, dtype=np.float32)
