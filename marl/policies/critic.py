"""
critic.py
Centralized Critic for MAPPO (CTDE).

Input:  global_state = concat of all agents' observations  (n_agents * obs_dim = 24)
Output: scalar value estimate  V(s)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(
        self,
        global_state_dim: int = 24,   # n_agents * obs_dim
        hidden_dim:       int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),       nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Last layer: small init for stable value estimates
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        global_state: (..., n_agents * obs_dim)
        Returns:      (..., 1)
        """
        return self.net(global_state)
