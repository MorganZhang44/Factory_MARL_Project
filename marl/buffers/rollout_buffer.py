"""
rollout_buffer.py
Fixed-size rollout buffer for MAPPO with GAE-Lambda advantage estimation.

Stores T steps × N agents of experience, then yields shuffled mini-batches.
"""
from __future__ import annotations

from typing import Generator
import numpy as np
import torch


class RolloutBuffer:
    def __init__(
        self,
        n_steps:       int,
        n_agents:      int,
        obs_dim:       int,
        action_dim:    int,
        gamma:         float = 0.99,
        gae_lambda:    float = 0.95,
        device:        str   = "cpu",
    ):
        self.n_steps    = n_steps
        self.n_agents   = n_agents
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.device     = device
        self.ptr        = 0
        self._allocate()

    def _allocate(self):
        T, N, O, A = self.n_steps, self.n_agents, self.obs_dim, self.action_dim
        self.observations = np.zeros((T, N, O), dtype=np.float32)
        self.actions      = np.zeros((T, N, A), dtype=np.float32)
        self.rewards      = np.zeros((T, N),    dtype=np.float32)
        self.values       = np.zeros((T, N),    dtype=np.float32)
        self.log_probs    = np.zeros((T, N),    dtype=np.float32)
        self.dones        = np.zeros(T,          dtype=np.float32)
        self.advantages   = np.zeros((T, N),    dtype=np.float32)
        self.returns      = np.zeros((T, N),    dtype=np.float32)

    def reset(self):
        self.ptr = 0
        self._allocate()

    @property
    def full(self) -> bool:
        return self.ptr >= self.n_steps

    def add(
        self,
        obs:      np.ndarray,   # (n_agents, obs_dim)
        action:   np.ndarray,   # (n_agents, action_dim)
        reward:   np.ndarray,   # (n_agents,)
        value:    np.ndarray,   # (n_agents,)
        log_prob: np.ndarray,   # (n_agents,)
        done:     float,
    ):
        t = self.ptr
        self.observations[t] = obs
        self.actions[t]      = action
        self.rewards[t]      = reward
        self.values[t]       = value
        self.log_probs[t]    = log_prob
        self.dones[t]        = done
        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        last_value: np.ndarray,   # (n_agents,) critic output at end-of-rollout
        last_done:  bool,
    ):
        """GAE-Lambda advantage estimation (in-place)."""
        gae = np.zeros(self.n_agents, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_done  = float(last_done)
                next_value = last_value
            else:
                next_done  = self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = (self.rewards[t]
                     + self.gamma * next_value * (1.0 - next_done)
                     - self.values[t])
            gae = delta + self.gamma * self.gae_lambda * (1.0 - next_done) * gae
            self.advantages[t] = gae
            self.returns[t]    = gae + self.values[t]

    def get_mini_batches(self, mini_batch_size: int) -> Generator:
        """
        Yields shuffled mini-batches as GPU/CPU tensors.
        Each batch: (obs, action, old_log_prob, advantage, return, global_state)
        """
        T, N = self.n_steps, self.n_agents
        total = T * N

        # Flatten (T, N, ...) → (T*N, ...)
        obs   = self.observations.reshape(total, self.obs_dim)
        acts  = self.actions.reshape(total, self.action_dim)
        logps = self.log_probs.reshape(total)
        advs  = self.advantages.reshape(total)
        rets  = self.returns.reshape(total)

        # Global state for critic: repeat each timestep's concat per agent
        # Shape: (T, N*O) → repeat N times along axis 0 → (T*N, N*O)
        gs_per_step = self.observations.reshape(T, N * self.obs_dim)   # (T, N*O)
        gs = np.repeat(gs_per_step, N, axis=0)                         # (T*N, N*O)

        # Normalize advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        indices = np.random.permutation(total)
        for start in range(0, total, mini_batch_size):
            idx = indices[start: start + mini_batch_size]
            to_t = lambda x: torch.FloatTensor(x[idx]).to(self.device)  # noqa: E731
            yield to_t(obs), to_t(acts), to_t(logps), to_t(advs), to_t(rets), to_t(gs)
