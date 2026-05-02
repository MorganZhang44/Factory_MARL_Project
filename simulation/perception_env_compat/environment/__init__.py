"""Environment package: Isaac scene, actors, and raw sensor output adapters.

The environment layer owns simulation-facing concerns only. It should publish
sensor packets and optional ground-truth diagnostics, but it should not run
perception or localization algorithms.
"""

from .types import (
    CameraSensorOutput,
    DogMotionHint,
    EnvironmentSensorFrame,
    GroundTruthState,
    ImuSensorOutput,
    LidarSensorOutput,
)

__all__ = [
    "CameraSensorOutput",
    "DogMotionHint",
    "EnvironmentSensorFrame",
    "GroundTruthState",
    "ImuSensorOutput",
    "LidarSensorOutput",
]
