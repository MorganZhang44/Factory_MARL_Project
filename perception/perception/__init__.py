# Copyright (c) 2026, Multi-Agent Surveillance Project
# Perception module - imported after AppLauncher initialization.
"""Perception package: sensor-frame input to global pose output."""

from .pipeline import PerceptionPipeline
from .types import DogPoseEstimate, IntruderPoseEstimate, PerceptionOutput

__all__ = [
    "DogPoseEstimate",
    "IntruderPoseEstimate",
    "PerceptionOutput",
    "PerceptionPipeline",
]
