"""
intruder_actor.py
Actor + Critic for the intruder evasion policy.

Mirrors marl/policies/actor.py and critic.py but with:
  - obs_dim = 20 (self pos+vel + 2 pursuers pos+vel + 8 lidar)
  - action_dim = 2 (relative offset, magnitude clamped to max_offset)
  - Centralized critic = single value head over the intruder's own obs
    (the intruder is a single agent — no need for CTDE)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class IntruderActor(nn.Module):
    def __init__(
        self,
        obs_dim:    int   = 20,
        action_dim: int   = 2,
        hidden_dim: int   = 64,
        max_offset: float = 2.0,
        log_std_init: float = -0.5,    # std=0.6 — keeps intruder from saturating its [-2,2] action clip
    ):
        super().__init__()
        self.max_offset = max_offset

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)

    def _dist(self, obs: torch.Tensor) -> Normal:
        feat = self.net(obs)
        mean = torch.tanh(self.mean_head(feat)) * self.max_offset
        std  = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def forward(self, obs: torch.Tensor) -> Normal:
        return self._dist(obs)

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        action = dist.mean if deterministic else dist.rsample()
        action = action.clamp(-self.max_offset, self.max_offset)
        log_p  = dist.log_prob(action).sum(dim=-1)
        return action, log_p

    def evaluate(
        self,
        obs:    torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist    = self._dist(obs)
        log_p   = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_p, entropy


class IntruderCritic(nn.Module):
    """Value head over the intruder's own observation (no CTDE needed for n=1)."""

    def __init__(
        self,
        obs_dim:    int = 20,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
