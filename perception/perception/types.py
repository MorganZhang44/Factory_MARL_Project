"""Typed perception outputs produced from environment sensor frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from environment.types import EnvironmentSensorFrame


@dataclass
class PerceiveRequest:
    """Wire-format request for `POST /perceive`.

    Wrapping the sensor frame in a request dataclass lets the simulation
    cheaply fan in to the perception server every step (for dog
    self-localization) without forcing the full intruder fusion pipeline
    to run on slim, camera-less frames.
    """

    frame: "EnvironmentSensorFrame"
    update_dogs: bool = True
    update_intruder: bool = True


@dataclass
class DogPoseEstimate:
    """Estimated and diagnostic state for one robot dog."""

    name: str
    step: int
    timestamp: float
    state: dict[str, Any]
    ground_truth: Any | None = None
    xy_error_m: float | None = None
    yaw_error_deg: float | None = None


@dataclass
class IntruderPoseEstimate:
    """Estimated and diagnostic state for the intruder/suspect."""

    step: int
    timestamp: float
    fusion_result: Any
    camera_detections: list[Any] = field(default_factory=list)
    lidar_detections: list[Any] = field(default_factory=list)
    ground_truth: Any | None = None


@dataclass
class PerceptionOutput:
    """Complete output of one perception update."""

    step: int
    timestamp: float
    dogs: dict[str, DogPoseEstimate] = field(default_factory=dict)
    intruder: IntruderPoseEstimate | None = None

    def pose_summary(self) -> dict[str, Any]:
        """Return a compact dict suitable for API layers or logging."""
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "dogs": {name: estimate.state for name, estimate in self.dogs.items()},
            "intruder": self.intruder.fusion_result if self.intruder is not None else None,
        }
