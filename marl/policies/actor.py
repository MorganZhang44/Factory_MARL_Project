"""
actor.py
Actor (policy) network for MAPPO.

Input:  local observation  (obs_dim = 13 in the current pursuit setup)
Output: Gaussian distribution over world-frame relative offset [dx, dy].

Parameter sharing: a single actor instance is used for ALL agents.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim:    int   = 13,
        action_dim: int   = 2,
        hidden_dim: int   = 64,
        map_half:   float = 10.0,
    ):
        super().__init__()
        self.map_half = map_half

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        # Learnable log-std (shared across action dims)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

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
        mean = torch.tanh(self.mean_head(feat)) * self.map_half  # → [-map_half, +map_half]
        std  = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def forward(self, obs: torch.Tensor) -> Normal:
        return self._dist(obs)

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return (action, log_prob)."""
        dist   = self._dist(obs)
        action = dist.mean if deterministic else dist.rsample()
        action = action.clamp(-self.map_half, self.map_half)
        log_p  = dist.log_prob(action).sum(dim=-1)
        return action, log_p

    def evaluate(
        self,
        obs:    torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (log_prob, entropy) for given (obs, action) pairs."""
        dist    = self._dist(obs)
        log_p   = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_p, entropy
