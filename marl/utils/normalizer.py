"""
normalizer.py
Online running mean/variance normalizer (Welford's algorithm).
Used to normalize observations before feeding to the network.
"""
from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Track running mean and variance of a fixed-shape vector."""

    def __init__(self, shape: tuple, epsilon: float = 1e-8):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = epsilon  # avoid div-by-zero on first update

    def update(self, x: np.ndarray) -> None:
        """x: single sample with shape == self.mean.shape, or batch (B, *shape)."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.mean.shape):
            x = x[np.newaxis]          # add batch dim
        batch_mean  = x.mean(axis=0)
        batch_var   = x.var(axis=0)
        batch_count = x.shape[0]

        total  = self.count + batch_count
        delta  = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_count / total)
        m_a  = self.var  * self.count
        m_b  = batch_var * batch_count
        new_var = (m_a + m_b + delta ** 2 * self.count * batch_count / total) / total

        self.mean  = new_mean
        self.var   = new_var
        self.count = total

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return np.clip(
            (x - self.mean.astype(np.float32)) / np.sqrt(self.var.astype(np.float32) + 1e-8),
            -clip, clip,
        )
