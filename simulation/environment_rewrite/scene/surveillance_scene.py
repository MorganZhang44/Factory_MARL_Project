# Copyright (c) 2026, Multi-Agent Surveillance Project
# Scene manager: smooth entity movement, ground truth extraction, trajectory tracking.

from __future__ import annotations

import math
import torch
import numpy as np
from typing import Sequence

from environment_rewrite.static_scene_geometry import (
    get_actor_spawn_points,
    get_surveillance_camera_names,
    plan_path_through_waypoints,
)


class SurveillanceSceneManager:
    """Manages the surveillance scene entities and their movement.

    Provides utilities for:
    - Smoothly moving dogs and suspect every simulation step
    - Generating predefined patrol / random walk paths
    - Extracting ground truth positions for validation
    - Tracking trajectories for visualization
    """

    def __init__(self, scene, sim_dt: float = 1.0 / 60.0):
        """Initialize the scene manager.

        Args:
            scene: The Isaac Lab InteractiveScene instance.
        """
        self.scene = scene
        self.sim_dt = float(sim_dt)

        scene_keys = self._scene_keys()

        # Entity references (required for movement logic)
        self.dog1 = self._get_entity("go2_dog_1") if "go2_dog_1" in scene_keys else None
        self.dog2 = self._get_entity("go2_dog_2") if "go2_dog_2" in scene_keys else None
        self.suspect = self._get_entity("suspect") if "suspect" in scene_keys else None

        # Camera references
        self.surveillance_cams = {}
        for cam_name in get_surveillance_camera_names():
            if cam_name in scene_keys:
                self.surveillance_cams[cam_name] = self._get_entity(cam_name)
        
        self.dog_cams = {}
        for cam_name in ["dog1_cam", "dog2_cam"]:
            if cam_name in scene_keys:
                self.dog_cams[cam_name] = self._get_entity(cam_name)

        # LiDAR references
        self.dog_lidars = {}
        for lidar_name in ["dog1_lidar", "dog2_lidar"]:
            if lidar_name in scene_keys:
                self.dog_lidars[lidar_name] = self._get_entity(lidar_name)

        spawns = get_actor_spawn_points()

        # Movement paths — full-map loops that exercise all CCTV views and
        # deliberately pass near obstacle islands to test partial occlusion.
        self._suspect_path = self._generate_suspect_path(start=spawns["suspect"])
        self._dog1_path = self._generate_dog_patrol_path(
            start=spawns["dog1"],
            clockwise=True,
            lane="outer",
        )
        self._dog2_path = self._generate_dog_patrol_path(
            start=spawns["dog2"],
            clockwise=False,
            lane="inner",
        )
        self._path_index = {"suspect": 0, "dog1": 0, "dog2": 0}
        self._motion_hints: dict[str, dict[str, tuple[float, ...]]] = {
            "go2_dog_1": {"linear_velocity": (0.0, 0.0, 0.0), "angular_velocity": (0.0, 0.0, 0.0)},
            "go2_dog_2": {"linear_velocity": (0.0, 0.0, 0.0), "angular_velocity": (0.0, 0.0, 0.0)},
        }

        # Speed control: how many sim steps per path point
        # Higher = slower movement and longer observation windows in the recordings.
        self._suspect_speed = 4
        self._dog_speed = 5

        # Trajectory tracking
        self._gt_trajectory: list[tuple[float, float, float]] = []
        self._est_trajectory: list[tuple[float, float, float | None]] = []

    def _scene_keys(self) -> set[str]:
        """Collect all currently registered scene keys."""
        keys = set()
        if hasattr(self.scene, "articulations"):
            keys.update(self.scene.articulations.keys())
        if hasattr(self.scene, "sensors"):
            keys.update(self.scene.sensors.keys())
        if hasattr(self.scene, "rigid_objects"):
            keys.update(self.scene.rigid_objects.keys())
        return keys

    def _get_entity(self, name: str):
        """Retrieve an entity from any available scene collection."""
        if hasattr(self.scene, "articulations") and name in self.scene.articulations.keys():
            return self.scene.articulations[name]
        if hasattr(self.scene, "sensors") and name in self.scene.sensors.keys():
            return self.scene.sensors[name]
        if hasattr(self.scene, "rigid_objects") and name in self.scene.rigid_objects.keys():
            return self.scene.rigid_objects[name]
        return None

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------

    def get_suspect_ground_truth(self) -> torch.Tensor:
        """Get the suspect's ground truth world position."""
        if self.suspect is None:
            return torch.zeros(3)
        return self.suspect.data.root_pos_w[0, :3].clone()

    def get_dog_positions(self) -> dict[str, torch.Tensor]:
        """Get both dogs' ground truth positions."""
        positions = {}
        if self.dog1 is not None:
            positions["dog1"] = self.dog1.data.root_pos_w[0, :3].clone()
        if self.dog2 is not None:
            positions["dog2"] = self.dog2.data.root_pos_w[0, :3].clone()
        return positions

    def get_all_positions(self) -> dict[str, torch.Tensor]:
        """Get all entity positions for logging.

        Returns:
            dict with keys 'suspect', 'dog1', 'dog2' → position tensors.
        """
        positions = self.get_dog_positions()
        positions["suspect"] = self.get_suspect_ground_truth()
        return positions

    # ------------------------------------------------------------------
    # Trajectory tracking
    # ------------------------------------------------------------------

    def record_ground_truth(self):
        """Record current ground truth position to trajectory."""
        gt = self.get_suspect_ground_truth()
        self._gt_trajectory.append((gt[0].item(), gt[1].item(), gt[2].item()))

    def record_estimation(self, est_pos):
        """Record estimated position to trajectory.

        Args:
            est_pos: Estimated position tensor (3,) or None.
        """
        if est_pos is not None:
            self._est_trajectory.append(
                (est_pos[0].item(), est_pos[1].item(), est_pos[2].item())
            )
        else:
            self._est_trajectory.append((None, None, None))

    def get_trajectories(self) -> dict:
        """Get recorded trajectories.

        Returns:
            {"ground_truth": [(x,y,z), ...], "estimated": [(x,y,z), ...]}
        """
        return {
            "ground_truth": self._gt_trajectory.copy(),
            "estimated": self._est_trajectory.copy(),
        }

    # ------------------------------------------------------------------
    # Teleportation (simulating movement)
    # ------------------------------------------------------------------

    def teleport_entity(
        self,
        entity,
        position: Sequence[float],
        orientation: Sequence[float] | None = None,
        linear_velocity: Sequence[float] | None = None,
        angular_velocity: Sequence[float] | None = None,
    ):
        """Teleport an articulation entity to a new position.

        Sets root pose, freezes all joints to default position, and zeros
        all velocities so the entity moves rigidly without limb twitching.

        Args:
            entity: The Articulation object.
            position: Target (x, y, z) position.
            orientation: Optional (w, x, y, z) quaternion. If None, keep current.
        """
        dev = entity.device

        # Root pose
        pos = torch.tensor([position], dtype=torch.float32, device=dev)
        if orientation is not None:
            quat = torch.tensor([orientation], dtype=torch.float32, device=dev)
        else:
            quat = entity.data.root_quat_w[:1].clone()

        entity.write_root_pose_to_sim(
            torch.cat([pos, quat], dim=-1)
        )

        if linear_velocity is None:
            linear_velocity = (0.0, 0.0, 0.0)
        if angular_velocity is None:
            angular_velocity = (0.0, 0.0, 0.0)
        root_velocity = torch.tensor(
            [[*linear_velocity, *angular_velocity]],
            dtype=torch.float32,
            device=dev,
        )
        entity.write_root_velocity_to_sim(root_velocity)

        # Freeze all joints: set to default position with zero velocity
        num_joints = entity.num_joints
        if num_joints > 0:
            # Default joint positions (the init state, typically all zeros)
            default_pos = entity.data.default_joint_pos[:1].clone()
            zero_joint_vel = torch.zeros(1, num_joints, dtype=torch.float32, device=dev)
            entity.write_joint_state_to_sim(default_pos, zero_joint_vel)

    def step_movement(self, step: int):
        """Advance all entities smoothly along their paths.

        Called every single simulation step for smooth motion.

        Args:
            step: Current simulation step number.
        """
        # Move suspect
        if self.suspect is not None and step % self._suspect_speed == 0:
            idx = self._path_index["suspect"]
            path = self._suspect_path
            pos = path[idx]

            # Face velocity direction
            next_idx = (idx + 1) % len(path)
            next_pos = path[next_idx]
            yaw = math.atan2(next_pos[1] - pos[1], next_pos[0] - pos[0])
            quat = self._yaw_to_quat(yaw)
            ctrl_dt = max(self._suspect_speed * self.sim_dt, 1.0e-6)
            vel = (
                (next_pos[0] - pos[0]) / ctrl_dt,
                (next_pos[1] - pos[1]) / ctrl_dt,
                (next_pos[2] - pos[2]) / ctrl_dt,
            )
            next_yaw = yaw
            if len(path) > 2:
                after_next_pos = path[(next_idx + 1) % len(path)]
                next_yaw = math.atan2(after_next_pos[1] - next_pos[1], after_next_pos[0] - next_pos[0])
            ang_vel = (0.0, 0.0, self._wrap_angle(next_yaw - yaw) / ctrl_dt)

            self.teleport_entity(self.suspect, pos, quat, linear_velocity=vel, angular_velocity=ang_vel)
            self._path_index["suspect"] = next_idx

        # Move dogs
        if step % self._dog_speed == 0:
            for path_key, dog in [("dog1", self.dog1), ("dog2", self.dog2)]:
                if dog is None:
                    continue
                idx = self._path_index[path_key]
                path = self._dog1_path if path_key == "dog1" else self._dog2_path
                pos = path[idx]

                next_idx = (idx + 1) % len(path)
                next_pos = path[next_idx]
                yaw = math.atan2(next_pos[1] - pos[1], next_pos[0] - pos[0])
                quat = self._yaw_to_quat(yaw)
                ctrl_dt = max(self._dog_speed * self.sim_dt, 1.0e-6)
                vel = (
                    (next_pos[0] - pos[0]) / ctrl_dt,
                    (next_pos[1] - pos[1]) / ctrl_dt,
                    (next_pos[2] - pos[2]) / ctrl_dt,
                )
                next_yaw = yaw
                if len(path) > 2:
                    after_next_pos = path[(next_idx + 1) % len(path)]
                    next_yaw = math.atan2(after_next_pos[1] - next_pos[1], after_next_pos[0] - next_pos[0])
                ang_vel = (0.0, 0.0, self._wrap_angle(next_yaw - yaw) / ctrl_dt)

                self.teleport_entity(dog, pos, quat, linear_velocity=vel, angular_velocity=ang_vel)
                dog_name = "go2_dog_1" if path_key == "dog1" else "go2_dog_2"
                self._motion_hints[dog_name] = {
                    "linear_velocity": vel,
                    "angular_velocity": ang_vel,
                }
                self._path_index[path_key] = next_idx

    def get_dog_motion_hints(self) -> dict[str, dict[str, tuple[float, ...]]]:
        """Return the latest commanded dog motion for local odometry prediction."""
        return {
            dog_name: {
                "linear_velocity": tuple(values["linear_velocity"]),
                "angular_velocity": tuple(values["angular_velocity"]),
            }
            for dog_name, values in self._motion_hints.items()
        }

    # ------------------------------------------------------------------
    # Path generation — high resolution for smooth movement
    # ------------------------------------------------------------------

    def _generate_suspect_path(self, start: tuple[float, float, float]) -> list[tuple[float, float, float]]:
        """Generate a smooth, high-resolution path for the suspect.

        Returns a densely interpolated list of waypoints.
        """
        anchors = [
            start,
            (4.75, 0.95, start[2]),
            (3.15, 2.45, start[2]),
            (0.15, 2.75, start[2]),
            (-4.25, 2.55, start[2]),
            (-4.55, 0.55, start[2]),
            (-4.10, -2.95, start[2]),
            (-0.70, -3.55, start[2]),
            (2.15, -3.35, start[2]),
            (4.20, -2.45, start[2]),
            (4.80, -0.25, start[2]),
            (2.35, -0.15, start[2]),
            (1.10, -2.15, start[2]),
            (-1.30, -2.20, start[2]),
            (-3.25, -1.95, start[2]),
            (-3.90, 1.55, start[2]),
            (-1.25, 2.35, start[2]),
            (1.90, 1.95, start[2]),
            start,
        ]
        planned = plan_path_through_waypoints(anchors, clearance=0.32, resolution=0.12)
        return self._interpolate_waypoints(planned, step_size=0.05)

    def _generate_dog_patrol_path(
        self,
        start: tuple[float, float, float],
        clockwise: bool,
        lane: str = "outer",
    ) -> list[tuple[float, float, float]]:
        """Generate a smooth whole-map patrol path for a dog."""
        outer_loop = [
            start,
            (-4.55, -3.45, start[2]),
            (-0.65, -3.75, start[2]),
            (2.65, -3.55, start[2]),
            (4.55, -2.55, start[2]),
            (4.95, -0.35, start[2]),
            (4.90, 1.00, start[2]),
            (3.00, 2.55, start[2]),
            (0.20, 2.75, start[2]),
            (-4.20, 2.55, start[2]),
            (-4.55, 0.55, start[2]),
            start,
        ]
        inner_loop = [
            start,
            (-3.65, 2.00, start[2]),
            (-0.95, 2.20, start[2]),
            (2.35, 1.95, start[2]),
            (4.10, 0.75, start[2]),
            (3.60, -1.95, start[2]),
            (1.55, -2.85, start[2]),
            (-1.10, -2.90, start[2]),
            (-3.35, -2.15, start[2]),
            (-4.05, -0.10, start[2]),
            start,
        ]
        route = outer_loop if lane == "outer" else inner_loop
        if not clockwise:
            route = [route[0], *list(reversed(route[1:-1])), route[0]]
        planned = plan_path_through_waypoints(route, clearance=0.30, resolution=0.12)
        return self._interpolate_waypoints(planned, step_size=0.055)

    @staticmethod
    def _interpolate_waypoints(
        waypoints: list[tuple[float, float, float]],
        step_size: float,
    ) -> list[tuple[float, float, float]]:
        """Linearly interpolate a closed waypoint path at fixed spatial resolution."""
        interpolated: list[tuple[float, float, float]] = []
        for i in range(len(waypoints) - 1):
            p0 = waypoints[i]
            p1 = waypoints[i + 1]
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            dz = p1[2] - p0[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            num_steps = max(1, int(dist / step_size))
            for t in range(num_steps):
                alpha = t / num_steps
                interpolated.append(
                    (
                        p0[0] + alpha * dx,
                        p0[1] + alpha * dy,
                        p0[2] + alpha * dz,
                    )
                )
        interpolated.append(waypoints[-1])
        return interpolated

    @staticmethod
    def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
        """Convert yaw angle to quaternion (w, x, y, z).

        Args:
            yaw: Yaw angle in radians.

        Returns:
            Quaternion as (w, x, y, z).
        """
        return (
            math.cos(yaw / 2),
            0.0,
            0.0,
            math.sin(yaw / 2),
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))
