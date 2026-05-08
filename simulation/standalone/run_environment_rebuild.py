#!/usr/bin/env python3
"""Phase-0 environment rebuild runtime.

This entrypoint is intentionally narrow. It exists to validate the clean scene
ownership and physics/contact assumptions before sensors and bridges are
reintroduced.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIMULATION_ROOT = PROJECT_ROOT / "simulation"
if str(SIMULATION_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATION_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment_rebuild.runtime_defaults import (  # noqa: E402
    CCTV_CAMERA_FOCAL_LENGTH,
    CCTV_CAMERA_HORIZONTAL_APERTURE,
    CCTV_CAMERA_NAMES,
    CCTV_HEIGHT,
    CCTV_PITCH_DEG,
    DEFAULT_SCENE_USD,
    DOGS,
    DOG_CAMERA_FOCAL_LENGTH,
    DOG_CAMERA_HORIZONTAL_APERTURE,
    DOG_CAMERA_POS,
    DOG_CAMERA_USD_ROT_WXYZ,
    FLOOR_PHYSICS_MATERIAL,
    FLOOR_PRIM_PATH,
    INTRUDER,
    INTRUDER_ARM_DOWN_POSE,
    INTRUDER_ROUTE_ANCHORS,
    INTRUDER_ROUTE_STEP_SIZE,
    INTRUDER_SPEED_MPS,
    LOCALIZATION_MESH_PATH,
    PERCEPTION_LIDAR_HORIZONTAL_FOV_DEG,
    PERCEPTION_LIDAR_MAX_DISTANCE,
    PERCEPTION_LIDAR_MOUNT_POS,
    PERCEPTION_LIDAR_VERTICAL_FOV_DEG,
    REFERENCED_PHYSICS_SCENE_PATH,
    ROBOT_PHYSICS_MATERIAL,
)


parser = argparse.ArgumentParser(description="Run the phase-0 environment rebuild runtime.")
parser.add_argument("--scene-usd", type=Path, default=DEFAULT_SCENE_USD, help="Static USDA/USD scene file.")
parser.add_argument("--steps", type=int, default=2000, help="Number of simulation steps to run.")
parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep.")
parser.add_argument("--keep-open", action="store_true", help="Keep Isaac Sim open after stepping.")
parser.add_argument(
    "--move-intruder",
    action="store_true",
    help="Move the intruder along its scripted route. Disabled by default for static-scene runs.",
)
parser.add_argument("--show-lidar", action="store_true", help="Enable RayCaster LiDAR debug visualization.")
parser.add_argument("--disable-ros2", action="store_true", help="Do not publish rebuild topics to ROS2.")
parser.add_argument("--topic-prefix", default="/factory/simulation", help="ROS2 topic prefix used by Core.")
parser.add_argument("--control-topic-prefix", default="/factory/control", help="ROS2 control topic prefix used by Core.")
parser.add_argument("--publish-every", type=int, default=4, help="Publish ROS2 data every N simulation steps.")
parser.add_argument("--max-command-age", type=float, default=1.0, help="Ignore stale control commands after this many seconds.")
parser.add_argument("--command-scale", type=float, default=1.0, help="Scale incoming world-frame velocity commands.")
parser.add_argument(
    "--loco-log-every",
    type=int,
    default=5,
    help="Print [LOCO] status every N simulation steps.",
)
parser.add_argument(
    "--view-camera",
    choices=["world"],
    default="world",
    help="Reserved for future rebuild camera options.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from pxr import Gf, Sdf, UsdGeom  # noqa: E402

try:
    import isaacsim.core.utils.prims as prim_utils  # noqa: E402
except ImportError:
    import omni.isaac.core.utils.prims as prim_utils  # type: ignore  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg, RayCaster, RayCasterCfg, patterns  # noqa: E402
from isaaclab_assets import HUMANOID_CFG, UNITREE_GO2_CFG  # noqa: E402
from environment_rewrite.localization_mesh import create_static_localization_mesh  # noqa: E402
from environment_rewrite.static_scene_geometry import get_camera_positions, get_camera_targets  # noqa: E402

try:  # noqa: E402
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Image, Imu, LaserScan, PointCloud2, PointField
    from std_msgs.msg import String
except ImportError:  # pragma: no cover
    rclpy = None
    PoseStamped = None
    Image = None
    Imu = None
    LaserScan = None
    PointCloud2 = None
    PointField = None
    String = None


class IsaacSimRos2Bridge:
    """Publish the rebuild runtime state using the legacy Simulation-Core ROS2 contract."""

    def __init__(
        self,
        topic_prefix: str,
        control_topic_prefix: str,
        robot_ids: list[str],
        intruder_ids: list[str],
        max_command_age: float,
    ) -> None:
        if rclpy is None:
            raise RuntimeError(
                "rclpy is unavailable. Launch through scripts/launch_simulation.sh "
                "so Isaac Sim's ROS2 bridge paths are configured."
            )
        rclpy.init(args=None)
        self.node = rclpy.create_node("factory_isaac_rebuild_publisher")
        self.topic_prefix = topic_prefix.rstrip("/")
        self.control_topic_prefix = control_topic_prefix.rstrip("/")
        self.robot_ids = robot_ids
        self.intruder_ids = intruder_ids
        self.cctv_ids = list(CCTV_CAMERA_NAMES)
        self.max_command_age = max_command_age
        self.motion_commands: dict[str, tuple[list[float], float]] = {}
        self.joint_action_commands: dict[str, tuple[list[float], float, float]] = {}

        self.state_pub = self.node.create_publisher(String, f"{self.topic_prefix}/state", 10)
        self.robot_pose_pubs = {
            robot_id: self.node.create_publisher(PoseStamped, f"{self.topic_prefix}/{robot_id}/pose", 20)
            for robot_id in robot_ids
        }
        self.intruder_pose_pubs = {
            intruder_id: self.node.create_publisher(PoseStamped, f"{self.topic_prefix}/{intruder_id}/pose", 20)
            for intruder_id in intruder_ids
        }
        self.camera_pubs = {
            robot_id: self.node.create_publisher(Image, f"{self.topic_prefix}/{robot_id}/camera/image_raw", 5)
            for robot_id in robot_ids
        }
        self.depth_pubs = {
            robot_id: self.node.create_publisher(Image, f"{self.topic_prefix}/{robot_id}/camera/depth", 5)
            for robot_id in robot_ids
        }
        self.semantic_pubs = {
            robot_id: self.node.create_publisher(
                Image, f"{self.topic_prefix}/{robot_id}/camera/semantic_segmentation", 5
            )
            for robot_id in robot_ids
        }
        self.imu_pubs = {
            robot_id: self.node.create_publisher(Imu, f"{self.topic_prefix}/{robot_id}/imu", 20)
            for robot_id in robot_ids
        }
        self.cctv_camera_pubs = {
            camera_id: self.node.create_publisher(Image, f"{self.topic_prefix}/cctv/{camera_id}/image_raw", 5)
            for camera_id in self.cctv_ids
        }
        self.cctv_semantic_pubs = {
            camera_id: self.node.create_publisher(
                Image, f"{self.topic_prefix}/cctv/{camera_id}/semantic_segmentation", 5
            )
            for camera_id in self.cctv_ids
        }
        self.lidar_pubs = {
            robot_id: self.node.create_publisher(LaserScan, f"{self.topic_prefix}/{robot_id}/lidar/scan", 10)
            for robot_id in robot_ids
        }
        self.lidar_point_cloud_pubs = {
            robot_id: self.node.create_publisher(PointCloud2, f"{self.topic_prefix}/{robot_id}/lidar/points", 5)
            for robot_id in robot_ids
        }
        self.locomotion_observation_pubs = {
            robot_id: self.node.create_publisher(String, f"{self.topic_prefix}/{robot_id}/locomotion/observation", 20)
            for robot_id in robot_ids
        }
        self.motion_command_sub = self.node.create_subscription(
            String,
            f"{self.control_topic_prefix}/locomotion/motion_command",
            self._on_motion_command,
            20,
        )
        print(f"[INFO] Rebuild ROS2 publisher active under {self.topic_prefix}")
        print(f"[INFO] Rebuild ROS2 control subscriber active under {self.control_topic_prefix}")

    def publish(
        self,
        dogs: dict[str, Articulation],
        intruder: Articulation,
        camera_readers: dict[str, Camera],
        cctv_readers: dict[str, Camera],
        lidar_readers: dict[str, RayCaster],
        step_idx: int,
    ) -> None:
        stamp = self.node.get_clock().now().to_msg()
        robot_states: dict[str, tuple[float, float, float]] = {}
        for robot_id in self.robot_ids:
            pos = _root_position(dogs[robot_id])
            robot_states[robot_id] = pos
            self.robot_pose_pubs[robot_id].publish(self._make_pose(pos, stamp, _root_quat(dogs[robot_id])))
            self.imu_pubs[robot_id].publish(self._make_imu(robot_id, stamp, dogs[robot_id]))
            self.locomotion_observation_pubs[robot_id].publish(
                String(data=json.dumps(_make_locomotion_observation_payload(robot_id, dogs[robot_id], step_idx)))
            )

            camera_msg = self._make_camera_image(robot_id, stamp, camera_readers.get(robot_id))
            if camera_msg is not None:
                self.camera_pubs[robot_id].publish(camera_msg)
            depth_msg = self._make_depth_image(robot_id, stamp, camera_readers.get(robot_id))
            if depth_msg is not None:
                self.depth_pubs[robot_id].publish(depth_msg)
            semantic_msg = self._make_semantic_image(robot_id, stamp, camera_readers.get(robot_id))
            if semantic_msg is not None:
                self.semantic_pubs[robot_id].publish(semantic_msg)
            lidar = lidar_readers.get(robot_id)
            scan_msg = self._make_scan(robot_id, stamp, lidar)
            if scan_msg is not None:
                self.lidar_pubs[robot_id].publish(scan_msg)
            point_cloud_msg = self._make_point_cloud(robot_id, stamp, lidar)
            if point_cloud_msg is not None:
                self.lidar_point_cloud_pubs[robot_id].publish(point_cloud_msg)

        robot_camera_infos = {
            robot_id: self._camera_info_payload(camera_readers.get(robot_id))
            for robot_id in self.robot_ids
        }

        for camera_id in self.cctv_ids:
            camera_msg = self._make_camera_image(camera_id, stamp, cctv_readers.get(camera_id))
            if camera_msg is not None:
                self.cctv_camera_pubs[camera_id].publish(camera_msg)
            semantic_msg = self._make_semantic_image(camera_id, stamp, cctv_readers.get(camera_id))
            if semantic_msg is not None:
                self.cctv_semantic_pubs[camera_id].publish(semantic_msg)

        cctv_camera_infos = {
            camera_id: self._camera_info_payload(cctv_readers.get(camera_id))
            for camera_id in self.cctv_ids
        }

        intruder_pos = _root_position(intruder)
        intruder_states = {"intruder_1": intruder_pos}
        self.intruder_pose_pubs["intruder_1"].publish(self._make_pose(intruder_pos, stamp, _root_quat(intruder)))
        self.state_pub.publish(
            String(
                data=json.dumps(
                    {
                        "timestamp": step_idx,
                        "frame_id": "world",
                        "robots": {robot_id: {"position": list(pos)} for robot_id, pos in robot_states.items()},
                        "intruders": {
                            intruder_id: {"position": list(pos)} for intruder_id, pos in intruder_states.items()
                        },
                        "camera_infos": {
                            "robots": robot_camera_infos,
                            "cctv": cctv_camera_infos,
                        },
                    }
                )
            )
        )

    def spin_once(self) -> None:
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def current_motion_commands(self) -> dict[str, list[float]]:
        now = time.monotonic()
        commands: dict[str, list[float]] = {}
        for robot_id, (velocity, updated_at) in self.motion_commands.items():
            if now - updated_at <= self.max_command_age:
                commands[robot_id] = velocity
        return commands

    def current_joint_actions(self) -> dict[str, tuple[list[float], float]]:
        now = time.monotonic()
        commands: dict[str, tuple[list[float], float]] = {}
        for robot_id, (action, action_scale, updated_at) in self.joint_action_commands.items():
            if now - updated_at <= self.max_command_age:
                commands[robot_id] = (action, action_scale)
        return commands

    def shutdown(self) -> None:
        self.node.destroy_node()
        rclpy.shutdown()

    def _on_motion_command(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            payload = data.get("payload", data)
            raw_commands = payload.get("commands")
            if isinstance(raw_commands, dict):
                for robot_id, command in raw_commands.items():
                    self._store_motion_command(str(robot_id), command.get("velocity"))
                    self._store_joint_action(str(robot_id), command.get("action"), command.get("action_scale"))
                return

            robot_id = payload.get("robot_id")
            velocity = payload.get("velocity")
            if robot_id is not None:
                self._store_motion_command(str(robot_id), velocity)
                self._store_joint_action(str(robot_id), payload.get("action"), payload.get("action_scale"))
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"[WARNING] Invalid motion command JSON: {exc}")

    def _store_motion_command(self, robot_id: str, velocity: Any) -> None:
        if robot_id not in self.robot_ids:
            return
        if not isinstance(velocity, (list, tuple)) or len(velocity) < 2:
            return
        vx = float(velocity[0])
        vy = float(velocity[1])
        if not math.isfinite(vx) or not math.isfinite(vy):
            return
        self.motion_commands[robot_id] = ([vx, vy], time.monotonic())

    def _store_joint_action(self, robot_id: str, action: Any, action_scale: Any) -> None:
        if robot_id not in self.robot_ids or action is None:
            return
        if not isinstance(action, (list, tuple)) or len(action) != 12:
            return
        values = [float(value) for value in action]
        if not all(math.isfinite(value) for value in values):
            return
        scale = 0.25 if action_scale is None else float(action_scale)
        self.joint_action_commands[robot_id] = (values, scale, time.monotonic())

    @staticmethod
    def _make_pose(
        pos: tuple[float, float, float],
        stamp,
        quat: tuple[float, float, float, float] | None = None,
    ) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "world"
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        quat = quat or (1.0, 0.0, 0.0, 0.0)
        msg.pose.orientation.w = float(quat[0])
        msg.pose.orientation.x = float(quat[1])
        msg.pose.orientation.y = float(quat[2])
        msg.pose.orientation.z = float(quat[3])
        return msg

    @staticmethod
    def _make_camera_image(sensor_id: str, stamp, camera: Camera | None) -> Image | None:
        if camera is None:
            return None
        rgb = camera.data.output.get("rgb")
        if rgb is None or rgb.numel() == 0:
            return None
        image = rgb[0].detach().cpu().numpy()
        if image.ndim != 3 or image.shape[2] < 3:
            return None
        image = image[:, :, :3]
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        height, width, _ = image.shape
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = f"{sensor_id}/front_camera" if sensor_id in DOGS else f"cctv/{sensor_id}"
        msg.height = height
        msg.width = width
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = width * 3
        msg.data = image.tobytes()
        return msg

    @staticmethod
    def _make_depth_image(robot_id: str, stamp, camera: Camera | None) -> Image | None:
        if camera is None:
            return None
        depth = camera.data.output.get("distance_to_image_plane")
        if depth is None or depth.numel() == 0:
            return None
        image = depth[0].detach().cpu().numpy()
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[:, :, 0]
        if image.ndim != 2:
            return None
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype("<f4", copy=False)
        height, width = image.shape
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = f"{robot_id}/front_camera"
        msg.height = height
        msg.width = width
        msg.encoding = "32FC1"
        msg.is_bigendian = False
        msg.step = width * 4
        msg.data = image.tobytes()
        return msg

    @staticmethod
    def _make_semantic_image(sensor_id: str, stamp, camera: Camera | None) -> Image | None:
        if camera is None:
            return None
        semantic = camera.data.output.get("semantic_segmentation")
        if semantic is None or semantic.numel() == 0:
            return None
        if semantic.dim() == 4:
            image = semantic[0, :, :, 0].detach().cpu().numpy()
        else:
            image = semantic[0].detach().cpu().numpy()
            if image.ndim == 3:
                image = image[:, :, 0] if image.shape[-1] == 1 else image[0]
        if image.ndim != 2:
            return None
        image = image.astype("<i4", copy=False)
        height, width = image.shape
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = f"{sensor_id}/front_camera" if sensor_id in DOGS else f"cctv/{sensor_id}"
        msg.height = height
        msg.width = width
        msg.encoding = "32SC1"
        msg.is_bigendian = False
        msg.step = width * 4
        msg.data = image.tobytes()
        return msg

    @staticmethod
    def _make_imu(robot_id: str, stamp, dog: Articulation) -> Imu:
        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = f"{robot_id}/base"
        quat = _root_quat(dog)
        msg.orientation.w = float(quat[0])
        msg.orientation.x = float(quat[1])
        msg.orientation.y = float(quat[2])
        msg.orientation.z = float(quat[3])
        ang_vel = dog.data.root_ang_vel_w[0].detach().cpu().tolist()
        msg.angular_velocity.x = float(ang_vel[0])
        msg.angular_velocity.y = float(ang_vel[1])
        msg.angular_velocity.z = float(ang_vel[2])
        try:
            lin_acc = dog.data.body_lin_acc_w[0, 0].detach().cpu().tolist()
        except (AttributeError, IndexError):
            lin_acc = [0.0, 0.0, 0.0]
        msg.linear_acceleration.x = float(lin_acc[0])
        msg.linear_acceleration.y = float(lin_acc[1])
        msg.linear_acceleration.z = float(lin_acc[2])
        return msg

    @staticmethod
    def _make_scan(robot_id: str, stamp, lidar: RayCaster | None) -> LaserScan | None:
        if lidar is None:
            return None
        points = _raycaster_points_sensor_frame(lidar)
        ranges = _derive_planar_scan_from_points(points)
        if ranges is None:
            return None
        msg = LaserScan()
        msg.header.stamp = stamp
        msg.header.frame_id = f"{robot_id}/front_lidar"
        msg.angle_min = -math.pi
        msg.angle_max = math.pi
        msg.angle_increment = math.radians(1.0)
        msg.time_increment = 0.0
        msg.scan_time = 0.1
        msg.range_min = 0.05
        msg.range_max = PERCEPTION_LIDAR_MAX_DISTANCE
        msg.ranges = ranges.astype(float).tolist()
        msg.intensities = [1.0 if math.isfinite(float(value)) else 0.0 for value in ranges]
        return msg

    @staticmethod
    def _make_point_cloud(robot_id: str, stamp, lidar: RayCaster | None) -> PointCloud2 | None:
        if lidar is None:
            return None
        points = _raycaster_points_sensor_frame(lidar)
        if points is None:
            return None
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = f"{robot_id}/front_lidar"
        msg.height = 1
        msg.width = int(points.shape[0])
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = False
        msg.data = points.astype("<f4", copy=False).tobytes()
        return msg

    @staticmethod
    def _camera_info_payload(camera: Camera | None) -> dict[str, Any]:
        if camera is None:
            return {}
        payload: dict[str, Any] = {}
        info = getattr(camera.data, "info", None)
        env_info = info[0] if isinstance(info, list) and len(info) > 0 else info
        if isinstance(env_info, dict):
            semantic_info = env_info.get("semantic_segmentation", {})
            if isinstance(semantic_info, dict):
                payload["info"] = IsaacSimRos2Bridge._json_safe_value(semantic_info)
        pos_w = getattr(camera.data, "pos_w", None)
        if pos_w is not None:
            try:
                payload["pos_w"] = IsaacSimRos2Bridge._json_safe_value(pos_w[0].detach().cpu().tolist())
            except (AttributeError, IndexError, TypeError):
                pass
        quat_w_ros = getattr(camera.data, "quat_w_ros", None)
        quat_w = quat_w_ros if quat_w_ros is not None else getattr(camera.data, "quat_w", None)
        if quat_w is not None:
            try:
                payload["quat_w"] = IsaacSimRos2Bridge._json_safe_value(quat_w[0].detach().cpu().tolist())
            except (AttributeError, IndexError, TypeError):
                pass
        intrinsic_matrices = getattr(camera.data, "intrinsic_matrices", None)
        if intrinsic_matrices is not None:
            try:
                payload["intrinsic_matrix"] = IsaacSimRos2Bridge._json_safe_value(
                    intrinsic_matrices[0].detach().cpu().tolist()
                )
            except (AttributeError, IndexError, TypeError):
                pass
        return payload

    @staticmethod
    def _json_safe_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): IsaacSimRos2Bridge._json_safe_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [IsaacSimRos2Bridge._json_safe_value(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value


def _interpolate_route(
    anchors: tuple[tuple[float, float, float], ...], step_size: float
) -> list[tuple[float, float, float]]:
    if len(anchors) < 2:
        return list(anchors)
    route: list[tuple[float, float, float]] = []
    for p0, p1 in zip(anchors[:-1], anchors[1:]):
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dz = p1[2] - p0[2]
        seg_len = math.sqrt(dx * dx + dy * dy + dz * dz)
        num_steps = max(1, int(math.ceil(seg_len / max(step_size, 1.0e-6))))
        for step_idx in range(num_steps):
            alpha = step_idx / num_steps
            route.append(
                (
                    p0[0] + alpha * dx,
                    p0[1] + alpha * dy,
                    p0[2] + alpha * dz,
                )
            )
    route.append(anchors[-1])
    return route


INTRUDER_ROUTE = _interpolate_route(INTRUDER_ROUTE_ANCHORS, INTRUDER_ROUTE_STEP_SIZE)
CCTV_POSITIONS = get_camera_positions(height=CCTV_HEIGHT)
CCTV_TARGETS = get_camera_targets()


def _yaw_quat_wxyz(yaw_deg: float) -> tuple[float, float, float, float]:
    yaw = math.radians(yaw_deg)
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def _quat_multiply(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _normalize_quat(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quat))
    if norm < 1.0e-8:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(value / norm for value in quat)


def _normalize(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    if norm < 1.0e-8:
        return (1.0, 0.0, 0.0)
    return (vec[0] / norm, vec[1] / norm, vec[2] / norm)


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _quat_from_rotmat(
    rot: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
) -> tuple[float, float, float, float]:
    r00, r01, r02 = rot[0]
    r10, r11, r12 = rot[1]
    r20, r21, r22 = rot[2]
    trace = r00 + r11 + r22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r21 - r12) / s
        y = (r02 - r20) / s
        z = (r10 - r01) / s
    elif r00 > r11 and r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        w = (r21 - r12) / s
        x = 0.25 * s
        y = (r01 + r10) / s
        z = (r02 + r20) / s
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        w = (r02 - r20) / s
        x = (r01 + r10) / s
        y = 0.25 * s
        z = (r12 + r21) / s
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        w = (r10 - r01) / s
        x = (r02 + r20) / s
        y = (r12 + r21) / s
        z = 0.25 * s
    return _normalize_quat((w, x, y, z))


def _look_at_with_fixed_pitch_quat(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    pitch_deg: float,
) -> tuple[float, float, float, float]:
    dx = target[0] - eye[0]
    dy = target[1] - eye[1]
    horizontal_norm = math.hypot(dx, dy)
    if horizontal_norm < 1.0e-6:
        return (1.0, 0.0, 0.0, 0.0)

    pitch_rad = math.radians(pitch_deg)
    forward = (
        math.cos(pitch_rad) * dx / horizontal_norm,
        math.cos(pitch_rad) * dy / horizontal_norm,
        -math.sin(pitch_rad),
    )
    world_up = (0.0, 0.0, 1.0)
    y_axis = _normalize(_cross(world_up, forward))
    z_axis = _normalize(_cross(forward, y_axis))
    rot = (
        (forward[0], y_axis[0], z_axis[0]),
        (forward[1], y_axis[1], z_axis[1]),
        (forward[2], y_axis[2], z_axis[2]),
    )
    return _quat_from_rotmat(rot)


def _set_xform(
    prim_path: str,
    translation: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
) -> None:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Cannot set transform. Prim does not exist: {prim_path}")
    sim_utils.standardize_xform_ops(
        prim,
        translation=tuple(float(value) for value in translation),
        orientation=tuple(float(value) for value in quat_wxyz),
        scale=(1.0, 1.0, 1.0),
    )


def _create_usd_camera(camera_path: str) -> None:
    stage = omni.usd.get_context().get_stage()
    camera = UsdGeom.Camera.Define(stage, Sdf.Path(camera_path))
    camera.CreateFocalLengthAttr(DOG_CAMERA_FOCAL_LENGTH)
    camera.CreateFocusDistanceAttr(400.0)
    camera.CreateHorizontalApertureAttr(DOG_CAMERA_HORIZONTAL_APERTURE)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
    _set_xform(camera_path, DOG_CAMERA_POS, quat_wxyz=DOG_CAMERA_USD_ROT_WXYZ)


def _create_cctv_camera(camera_path: str, camera_id: str) -> None:
    stage = omni.usd.get_context().get_stage()
    camera = UsdGeom.Camera.Define(stage, Sdf.Path(camera_path))
    camera.CreateFocalLengthAttr(CCTV_CAMERA_FOCAL_LENGTH)
    camera.CreateFocusDistanceAttr(400.0)
    camera.CreateHorizontalApertureAttr(CCTV_CAMERA_HORIZONTAL_APERTURE)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
    position = CCTV_POSITIONS[camera_id]
    target = CCTV_TARGETS[camera_id]
    world_quat = _look_at_with_fixed_pitch_quat(position, target, CCTV_PITCH_DEG)
    usd_quat = _normalize_quat(_quat_multiply(world_quat, DOG_CAMERA_USD_ROT_WXYZ))
    _set_xform(camera_path, position, quat_wxyz=usd_quat)


def _load_environment(scene_usd: Path) -> None:
    if not scene_usd.exists():
        raise FileNotFoundError(f"Scene file does not exist: {scene_usd}")
    prim_utils.create_prim("/World", "Xform")
    prim_utils.create_prim("/World/SlamScene", "Xform", usd_path=str(scene_usd))
    prim_utils.create_prim("/World/Actors", "Xform")


def _deactivate_referenced_physics_scene() -> None:
    stage = omni.usd.get_context().get_stage()
    physics_scene_prim = stage.GetPrimAtPath(REFERENCED_PHYSICS_SCENE_PATH)
    if physics_scene_prim.IsValid() and physics_scene_prim.IsActive():
        physics_scene_prim.SetActive(False)
        print(f"[INFO] Deactivated referenced PhysicsScene at {REFERENCED_PHYSICS_SCENE_PATH}")
    else:
        print(f"[INFO] No active referenced PhysicsScene at {REFERENCED_PHYSICS_SCENE_PATH}")


def _bind_physics_material(target_prim_path: str, material_cfg_dict: dict[str, object]) -> None:
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=float(material_cfg_dict["static_friction"]),
        dynamic_friction=float(material_cfg_dict["dynamic_friction"]),
        restitution=float(material_cfg_dict["restitution"]),
        friction_combine_mode=str(material_cfg_dict.get("friction_combine_mode", "average")),
        restitution_combine_mode=str(material_cfg_dict.get("restitution_combine_mode", "average")),
    )
    material_path = str(material_cfg_dict["prim_path"])
    material_cfg.func(material_path, material_cfg)
    sim_utils.bind_physics_material(target_prim_path, material_path)


def _apply_floor_physics_material() -> None:
    _bind_physics_material(FLOOR_PRIM_PATH, FLOOR_PHYSICS_MATERIAL)
    print(f"[INFO] Bound floor physics material at {FLOOR_PRIM_PATH}")


def _spawn_actors() -> tuple[dict[str, Articulation], Articulation]:
    dogs: dict[str, Articulation] = {}
    for robot_id, spec in DOGS.items():
        cfg = UNITREE_GO2_CFG.replace(prim_path=spec["prim_path"])
        cfg.spawn.semantic_tags = [("class", "dog")]
        dogs[robot_id] = Articulation(cfg)

    intruder_cfg = HUMANOID_CFG.replace(prim_path=INTRUDER["prim_path"])
    intruder_cfg.spawn.semantic_tags = [("class", "suspect")]
    intruder = Articulation(intruder_cfg)
    return dogs, intruder


def _attach_sensors(dog_specs: dict[str, dict]) -> tuple[list[str], dict[str, str]]:
    sensor_paths: list[str] = []
    lidar_parent_paths: dict[str, str] = {}
    for dog_id, spec in dog_specs.items():
        base_path = f"{spec['prim_path']}/base"
        camera_path = f"{base_path}/front_camera"
        _create_usd_camera(camera_path)
        lidar_parent_paths[dog_id] = base_path
        sensor_paths.append(camera_path)
        print(f"[INFO] {dog_id} camera: {camera_path}")
        print(f"[INFO] {dog_id} RayCaster LiDAR parent: {base_path}")
    return sensor_paths, lidar_parent_paths


def _attach_cctv_sensors() -> list[str]:
    prim_utils.create_prim("/World/CCTV", "Xform")
    camera_paths: list[str] = []
    for camera_id in CCTV_CAMERA_NAMES:
        camera_path = f"/World/CCTV/{camera_id}"
        _create_cctv_camera(camera_path, camera_id)
        camera_paths.append(camera_path)
        print(f"[INFO] CCTV camera {camera_id}: {camera_path}")
    return camera_paths


def _create_camera_readers(dog_specs: dict[str, dict]) -> dict[str, Camera]:
    cameras: dict[str, Camera] = {}
    for dog_id, spec in dog_specs.items():
        camera_path = f"{spec['prim_path']}/base/front_camera"
        cfg = CameraCfg(
            prim_path=camera_path,
            spawn=None,
            width=320,
            height=240,
            data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
            colorize_semantic_segmentation=False,
            update_period=0.0,
        )
        cameras[dog_id] = Camera(cfg)
        print(f"[INFO] {dog_id} RGB camera reader: {camera_path}")
    return cameras


def _create_cctv_readers() -> dict[str, Camera]:
    cameras: dict[str, Camera] = {}
    for camera_id in CCTV_CAMERA_NAMES:
        camera_path = f"/World/CCTV/{camera_id}"
        cfg = CameraCfg(
            prim_path=camera_path,
            spawn=None,
            width=640,
            height=480,
            data_types=["rgb", "semantic_segmentation"],
            colorize_semantic_segmentation=False,
            update_period=0.0,
        )
        cameras[camera_id] = Camera(cfg)
        print(f"[INFO] CCTV RGB camera reader: {camera_path}")
    return cameras


def _create_lidar_readers(lidar_parent_paths: dict[str, str], show_visualization: bool = False) -> dict[str, RayCaster]:
    lidar_readers: dict[str, RayCaster] = {}
    for dog_id, parent_path in lidar_parent_paths.items():
        cfg = RayCasterCfg(
            prim_path=parent_path,
            ray_alignment="base",
            pattern_cfg=patterns.LidarPatternCfg(
                channels=16,
                vertical_fov_range=PERCEPTION_LIDAR_VERTICAL_FOV_DEG,
                horizontal_fov_range=PERCEPTION_LIDAR_HORIZONTAL_FOV_DEG,
                horizontal_res=1.0,
            ),
            offset=RayCasterCfg.OffsetCfg(pos=PERCEPTION_LIDAR_MOUNT_POS),
            debug_vis=show_visualization,
            max_distance=PERCEPTION_LIDAR_MAX_DISTANCE,
            mesh_prim_paths=[LOCALIZATION_MESH_PATH],
        )
        lidar = RayCaster(cfg=cfg)
        lidar_readers[dog_id] = lidar
        print(f"[INFO] {dog_id} RayCaster LiDAR: parent={parent_path}, mesh={LOCALIZATION_MESH_PATH}")
    return lidar_readers


def _update_camera_readers(camera_readers: dict[str, Camera], dt: float) -> None:
    for camera in camera_readers.values():
        camera.update(dt)


def _update_lidar_readers(lidar_readers: dict[str, RayCaster], dt: float) -> None:
    for lidar in lidar_readers.values():
        lidar.update(dt, force_recompute=True)


def _reset_sensor_readers(camera_readers: dict[str, Camera], lidar_readers: dict[str, RayCaster]) -> None:
    for camera in camera_readers.values():
        if hasattr(camera, "reset"):
            camera.reset()
    for lidar in lidar_readers.values():
        if hasattr(lidar, "reset"):
            lidar.reset()


def _camera_frame_shape(camera: Camera | None, key: str) -> tuple[int, ...] | None:
    if camera is None:
        return None
    tensor = camera.data.output.get(key)
    if tensor is None or tensor.numel() == 0:
        return None
    return tuple(int(x) for x in tensor.shape)


def _lidar_point_count(lidar: RayCaster | None) -> int:
    if lidar is None:
        return 0
    hits = lidar.data.ray_hits_w
    if hits is None or hits.numel() == 0:
        return 0
    return int(hits[0].shape[0])


def _apply_robot_physics_material() -> None:
    for robot_id, spec in DOGS.items():
        _bind_physics_material(spec["prim_path"], ROBOT_PHYSICS_MATERIAL)
        print(f"[INFO] Bound robot physics material for {robot_id} at {spec['prim_path']}")


def _initialize_actor_states(dogs: dict[str, Articulation], intruder: Articulation) -> None:
    for robot_id, spec in DOGS.items():
        dog = dogs[robot_id]
        root_state = dog.data.default_root_state.clone()
        root_state[:, 0] = float(spec["pos"][0])
        root_state[:, 1] = float(spec["pos"][1])
        root_state[:, 2] = float(spec["pos"][2])
        quat = _yaw_quat_wxyz(float(spec["yaw_deg"]))
        root_state[:, 3] = quat[0]
        root_state[:, 4] = quat[1]
        root_state[:, 5] = quat[2]
        root_state[:, 6] = quat[3]
        dog.write_root_state_to_sim(root_state)
        dog.write_joint_state_to_sim(dog.data.default_joint_pos, dog.data.default_joint_vel)

    root_state = intruder.data.default_root_state.clone()
    root_state[:, 0] = float(INTRUDER["pos"][0])
    root_state[:, 1] = float(INTRUDER["pos"][1])
    root_state[:, 2] = float(INTRUDER["pos"][2])
    quat = _yaw_quat_wxyz(float(INTRUDER["yaw_deg"]))
    root_state[:, 3] = quat[0]
    root_state[:, 4] = quat[1]
    root_state[:, 5] = quat[2]
    root_state[:, 6] = quat[3]
    intruder.write_root_state_to_sim(root_state)
    intruder.write_joint_state_to_sim(intruder.data.default_joint_pos, intruder.data.default_joint_vel)


def _move_intruder_along_route(intruder: Articulation, step_idx: int, dt: float) -> None:
    if not INTRUDER_ROUTE:
        return
    route_len = len(INTRUDER_ROUTE)
    route_step_interval = max(1, int(round(INTRUDER_ROUTE_STEP_SIZE / max(INTRUDER_SPEED_MPS * dt, 1.0e-6))))
    idx = (step_idx // route_step_interval) % route_len
    next_idx = (idx + 1) % route_len
    pos = INTRUDER_ROUTE[idx]
    next_pos = INTRUDER_ROUTE[next_idx]
    dx = next_pos[0] - pos[0]
    dy = next_pos[1] - pos[1]
    dz = next_pos[2] - pos[2]
    yaw = math.atan2(dy, dx) if abs(dx) > 1.0e-9 or abs(dy) > 1.0e-9 else math.radians(INTRUDER["yaw_deg"])
    quat_w = math.cos(0.5 * yaw)
    quat_z = math.sin(0.5 * yaw)
    ctrl_dt = max(route_step_interval * dt, 1.0e-6)
    vx = dx / ctrl_dt
    vy = dy / ctrl_dt
    vz = dz / ctrl_dt

    root_state = intruder.data.root_state_w.clone()
    root_state[:, 0] = float(pos[0])
    root_state[:, 1] = float(pos[1])
    root_state[:, 2] = float(pos[2])
    root_state[:, 3] = quat_w
    root_state[:, 4] = 0.0
    root_state[:, 5] = 0.0
    root_state[:, 6] = quat_z
    root_state[:, 7] = vx
    root_state[:, 8] = vy
    root_state[:, 9] = vz
    root_state[:, 10] = 0.0
    root_state[:, 11] = 0.0
    root_state[:, 12] = 0.0
    intruder.write_root_state_to_sim(root_state)


def _capture_intruder_locked_pose(intruder: Articulation) -> None:
    joint_pos = intruder.data.joint_pos.clone()
    debug_lines: list[str] = []
    for joint_name_key, value in INTRUDER_ARM_DOWN_POSE.items():
        joint_ids = [idx for idx, joint_name in enumerate(intruder.joint_names) if joint_name == joint_name_key]
        if not joint_ids:
            debug_lines.append(f"{joint_name_key}: <no matches>")
            continue
        for joint_id in joint_ids:
            joint_name = intruder.joint_names[joint_id]
            old_value = float(joint_pos[0, joint_id].item())
            joint_pos[:, joint_id] = float(value)
            debug_lines.append(
                f"{joint_name_key}: joint_id={joint_id} name={joint_name} old={old_value:.4f} target={float(value):.4f}"
            )
    intruder._factory_locked_joint_pos = joint_pos
    intruder._factory_locked_joint_vel = torch.zeros_like(intruder.data.joint_vel)
    if not getattr(intruder, "_factory_arm_pose_debug_printed", False):
        print("[INFO] Intruder arm pose overrides:")
        for line in debug_lines:
            print(f"[INFO]   {line}")
        intruder._factory_arm_pose_debug_printed = True


def _lock_intruder_joint_pose(intruder: Articulation) -> None:
    joint_pos = getattr(intruder, "_factory_locked_joint_pos", None)
    joint_vel = getattr(intruder, "_factory_locked_joint_vel", None)
    if joint_pos is None or joint_vel is None:
        joint_pos = intruder.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(intruder.data.default_joint_vel)
    intruder.write_joint_state_to_sim(joint_pos, joint_vel)
    intruder.set_joint_position_target(joint_pos)


def _validate_prims() -> None:
    stage = omni.usd.get_context().get_stage()
    expected = [
        "/World/SlamScene",
        "/World/Actors",
        FLOOR_PRIM_PATH,
        *[spec["prim_path"] for spec in DOGS.values()],
        INTRUDER["prim_path"],
    ]
    missing = [path for path in expected if not stage.GetPrimAtPath(path).IsValid()]
    if missing:
        raise RuntimeError(f"Missing expected prims: {missing}")


def _root_position(entity: Articulation) -> tuple[float, float, float]:
    pos = entity.data.root_pos_w[0].detach().cpu().tolist()
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def _root_quat(entity: Articulation) -> tuple[float, float, float, float]:
    quat = entity.data.root_quat_w[0].detach().cpu().tolist()
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def _rotate_points_by_inverse_quat(points: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(value) for value in quat_wxyz]
    rot = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return points @ rot


def _raycaster_points_sensor_frame(lidar: RayCaster) -> np.ndarray | None:
    hits_w = lidar.data.ray_hits_w
    pos_w = lidar.data.pos_w
    quat_w = lidar.data.quat_w
    if hits_w is None or pos_w is None or quat_w is None:
        return None
    if hits_w.numel() == 0:
        return None

    hits = hits_w[0].detach().cpu().numpy().astype(np.float32, copy=False)
    origin = pos_w[0].detach().cpu().numpy().astype(np.float32, copy=False)
    quat = quat_w[0].detach().cpu().numpy().astype(np.float32, copy=False)
    points = _rotate_points_by_inverse_quat(hits - origin, quat)
    ranges = np.linalg.norm(points, axis=1)
    valid = np.isfinite(points).all(axis=1) & (ranges <= PERCEPTION_LIDAR_MAX_DISTANCE)
    points = points[valid]
    return points.astype(np.float32) if points.size else None


def _derive_planar_scan_from_points(points: np.ndarray | None) -> np.ndarray | None:
    if points is None or points.size == 0:
        return None

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    xy_ranges = np.linalg.norm(points[:, :2], axis=1)
    angles = np.arctan2(points[:, 1], points[:, 0])
    angle_min = -math.pi
    angle_max = math.pi
    angle_increment = math.radians(1.0)
    valid = (
        np.isfinite(xy_ranges)
        & np.isfinite(angles)
        & (xy_ranges >= 0.05)
        & (xy_ranges <= PERCEPTION_LIDAR_MAX_DISTANCE)
        & (angles >= angle_min)
        & (angles <= angle_max)
    )
    if not valid.any():
        return None

    bin_count = int(round((angle_max - angle_min) / angle_increment)) + 1
    ranges = np.full((bin_count,), np.inf, dtype=np.float32)
    bin_indices = np.floor((angles[valid] - angle_min) / angle_increment).astype(np.int32)
    bin_indices = np.clip(bin_indices, 0, bin_count - 1)
    np.minimum.at(ranges, bin_indices, xy_ranges[valid].astype(np.float32))
    return ranges


def _tensor_1d(entity: Articulation, name: str, size: int, default: float = 0.0) -> list[float]:
    value = getattr(entity.data, name, None)
    if value is None:
        return [default] * size
    data = value[0].detach().cpu().tolist()
    return [float(item) for item in data[:size]]


def _make_locomotion_observation(dog: Articulation) -> np.ndarray:
    base_lin_vel = _tensor_1d(dog, "root_lin_vel_b", 3)
    base_ang_vel = _tensor_1d(dog, "root_ang_vel_b", 3)
    projected_gravity = _tensor_1d(dog, "projected_gravity_b", 3, default=0.0)
    if projected_gravity == [0.0, 0.0, 0.0]:
        projected_gravity = [0.0, 0.0, -1.0]

    joint_pos = dog.data.joint_pos[0]
    joint_vel = dog.data.joint_vel[0]
    default_joint_pos = dog.data.default_joint_pos[0]
    default_joint_vel = dog.data.default_joint_vel[0]
    joint_pos_rel = (joint_pos - default_joint_pos).detach().cpu().tolist()
    joint_vel_rel = (joint_vel - default_joint_vel).detach().cpu().tolist()
    last_action = getattr(dog, "_factory_last_low_level_action", [0.0] * 12)

    observation = (
        base_lin_vel
        + base_ang_vel
        + projected_gravity
        + [0.0, 0.0, 0.0]
        + [float(value) for value in joint_pos_rel[:12]]
        + [float(value) for value in joint_vel_rel[:12]]
        + [float(value) for value in last_action[:12]]
    )
    return np.asarray(observation, dtype=np.float32)


def _make_locomotion_observation_payload(robot_id: str, dog: Articulation, step_idx: int) -> dict[str, Any]:
    return {
        "robot_id": robot_id,
        "timestamp": step_idx,
        "observation": _make_locomotion_observation(dog).tolist(),
        "schema": "go2_flat_velocity_policy_obs_v1",
    }


def _apply_motion_commands(
    dogs: dict[str, Articulation],
    commands: dict[str, list[float]],
    joint_actions: dict[str, tuple[list[float], float]],
    dt: float,
    command_scale: float,
) -> set[str]:
    low_level_applied: set[str] = set()
    for robot_id, (action, action_scale) in joint_actions.items():
        dog = dogs.get(robot_id)
        if dog is None:
            continue
        action_tensor = torch.tensor(
            action, device=dog.data.joint_pos.device, dtype=dog.data.joint_pos.dtype
        ).unsqueeze(0)
        target = dog.data.default_joint_pos + action_tensor * float(action_scale)
        dog.set_joint_position_target(target)
        dog._factory_last_low_level_action = list(action)
        low_level_applied.add(robot_id)

    for robot_id, velocity in commands.items():
        if robot_id in low_level_applied:
            continue
        dog = dogs.get(robot_id)
        if dog is None:
            continue
        vx = float(velocity[0]) * command_scale
        vy = float(velocity[1]) * command_scale
        root_state = dog.data.root_state_w.clone()
        root_state[:, 0] += vx * dt
        root_state[:, 1] += vy * dt
        if root_state.shape[1] >= 10:
            root_state[:, 7] = vx
            root_state[:, 8] = vy
            root_state[:, 9] = 0.0
        dog.write_root_state_to_sim(root_state)
    return low_level_applied


def _print_locomotion_status(
    dogs: dict[str, Articulation],
    commands: dict[str, list[float]],
    step_idx: int,
) -> None:
    for robot_id, dog in dogs.items():
        observed_body = _tensor_1d(dog, "root_lin_vel_b", 3)
        planar_speed = math.hypot(observed_body[0], observed_body[1])
        command = commands.get(robot_id, [0.0, 0.0])
        print(
            f"[LOCO] step={step_idx} {robot_id} "
            f"command=[{float(command[0]):.3f}, {float(command[1]):.3f}, 0.000] "
            f"observed_body=[{observed_body[0]:.3f}, {observed_body[1]:.3f}, {observed_body[2]:.3f}] "
            f"planar_speed={planar_speed:.3f}"
        )


def main() -> None:
    try:
        print(f"[INFO] Rebuild runtime loading scene: {args_cli.scene_usd}")
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args_cli.dt, device=args_cli.device))
        sim.set_camera_view(eye=[6.0, -7.0, 5.0], target=[0.0, -0.5, 0.5])

        _load_environment(args_cli.scene_usd)
        _deactivate_referenced_physics_scene()
        _apply_floor_physics_material()
        mesh_info = create_static_localization_mesh(LOCALIZATION_MESH_PATH)
        print(f"[INFO] Localization mesh: {mesh_info}")
        dogs, intruder = _spawn_actors()
        _apply_robot_physics_material()
        sensor_paths, lidar_parent_paths = _attach_sensors(DOGS)
        cctv_paths = _attach_cctv_sensors()
        camera_readers = _create_camera_readers(DOGS)
        cctv_readers = _create_cctv_readers()
        lidar_readers = _create_lidar_readers(lidar_parent_paths, show_visualization=args_cli.show_lidar)
        _validate_prims()

        sim.reset()
        _reset_sensor_readers(camera_readers, lidar_readers)
        _initialize_actor_states(dogs, intruder)
        sim.step()
        for dog in dogs.values():
            dog.update(args_cli.dt)
        intruder.update(args_cli.dt)
        _update_camera_readers(camera_readers, args_cli.dt)
        _update_camera_readers(cctv_readers, args_cli.dt)
        _update_lidar_readers(lidar_readers, args_cli.dt)
        _capture_intruder_locked_pose(intruder)

        # Keep the floor invisible to the debug camera only if desired later.
        floor_prim = omni.usd.get_context().get_stage().GetPrimAtPath(FLOOR_PRIM_PATH)
        if floor_prim.IsValid():
            UsdGeom.Imageable(floor_prim).MakeVisible()

        ros2_bridge = None if args_cli.disable_ros2 else IsaacSimRos2Bridge(
            topic_prefix=args_cli.topic_prefix,
            control_topic_prefix=args_cli.control_topic_prefix,
            robot_ids=list(DOGS.keys()),
            intruder_ids=["intruder_1"],
            max_command_age=args_cli.max_command_age,
        )
        omni.timeline.get_timeline_interface().play()
        print("[INFO] Rebuild runtime initialized. Stepping simulation...")
        wall_clock_start = time.monotonic()
        for step_idx in range(args_cli.steps):
            low_level_applied: set[str] = set()
            current_motion_commands: dict[str, list[float]] = {}
            if ros2_bridge is not None:
                ros2_bridge.spin_once()
                current_motion_commands = ros2_bridge.current_motion_commands()
                low_level_applied = _apply_motion_commands(
                    dogs,
                    current_motion_commands,
                    ros2_bridge.current_joint_actions(),
                    args_cli.dt,
                    args_cli.command_scale,
                )
            for dog_id, dog in dogs.items():
                if dog_id in low_level_applied:
                    dog.write_data_to_sim()
                    continue
                dog.write_data_to_sim()
            if args_cli.move_intruder:
                _move_intruder_along_route(intruder, step_idx, args_cli.dt)
            _lock_intruder_joint_pose(intruder)
            intruder.write_data_to_sim()
            sim.step()
            for dog in dogs.values():
                dog.update(args_cli.dt)
            intruder.update(args_cli.dt)
            _update_camera_readers(camera_readers, args_cli.dt)
            _update_camera_readers(cctv_readers, args_cli.dt)
            _update_lidar_readers(lidar_readers, args_cli.dt)
            if step_idx in (0, args_cli.steps - 1) or (step_idx > 0 and step_idx % max(1, args_cli.loco_log_every) == 0):
                _print_locomotion_status(dogs, current_motion_commands, step_idx)
            if step_idx in (0, args_cli.steps - 1) or (step_idx > 0 and step_idx % 100 == 0):
                for robot_id, camera in camera_readers.items():
                    print(
                        f"[SENSOR] step={step_idx} {robot_id} "
                        f"rgb={_camera_frame_shape(camera, 'rgb')} "
                        f"depth={_camera_frame_shape(camera, 'distance_to_image_plane')} "
                        f"semantic={_camera_frame_shape(camera, 'semantic_segmentation')} "
                        f"lidar_points={_lidar_point_count(lidar_readers.get(robot_id))}"
                    )
                for camera_id, camera in cctv_readers.items():
                    print(
                        f"[SENSOR] step={step_idx} {camera_id} "
                        f"rgb={_camera_frame_shape(camera, 'rgb')} "
                        f"semantic={_camera_frame_shape(camera, 'semantic_segmentation')}"
                    )
            if ros2_bridge is not None and step_idx % max(1, args_cli.publish_every) == 0:
                ros2_bridge.publish(dogs, intruder, camera_readers, cctv_readers, lidar_readers, step_idx)
            if step_idx > 0 and step_idx % 200 == 0:
                sim_time = (step_idx + 1) * args_cli.dt
                wall_time = time.monotonic() - wall_clock_start
                rtf = sim_time / wall_time if wall_time > 1.0e-6 else 0.0
                print(
                    f"[TIMING] step={step_idx}  sim_time={sim_time:.3f}s  "
                    f"wall_time={wall_time:.3f}s  RTF={rtf:.4f}  "
                    f"(sim is {'faster' if rtf > 1.0 else 'slower'} than real-time by {abs(1.0 - rtf)*100:.1f}%)"
                )

        print("[INFO] Rebuild runtime step loop finished.")
        if args_cli.keep_open:
            print("[INFO] keep-open enabled; leaving Isaac Sim running.")
            keep_open_step = args_cli.steps
            wall_clock_start = time.monotonic()
            while simulation_app.is_running():
                low_level_applied = set()
                current_motion_commands = {}
                if ros2_bridge is not None:
                    ros2_bridge.spin_once()
                    current_motion_commands = ros2_bridge.current_motion_commands()
                    low_level_applied = _apply_motion_commands(
                        dogs,
                        current_motion_commands,
                        ros2_bridge.current_joint_actions(),
                        args_cli.dt,
                        args_cli.command_scale,
                    )
                for dog_id, dog in dogs.items():
                    if dog_id in low_level_applied:
                        dog.write_data_to_sim()
                        continue
                    dog.write_data_to_sim()
                if args_cli.move_intruder:
                    _move_intruder_along_route(intruder, keep_open_step, args_cli.dt)
                _lock_intruder_joint_pose(intruder)
                intruder.write_data_to_sim()
                sim.step()
                for dog in dogs.values():
                    dog.update(args_cli.dt)
                intruder.update(args_cli.dt)
                _update_camera_readers(camera_readers, args_cli.dt)
                _update_camera_readers(cctv_readers, args_cli.dt)
                _update_lidar_readers(lidar_readers, args_cli.dt)
                if ros2_bridge is not None and keep_open_step % max(1, args_cli.publish_every) == 0:
                    ros2_bridge.publish(dogs, intruder, camera_readers, cctv_readers, lidar_readers, keep_open_step)
                keep_open_step += 1
                if keep_open_step % max(1, args_cli.loco_log_every) == 0:
                    _print_locomotion_status(dogs, current_motion_commands, keep_open_step)
                if keep_open_step > 0 and keep_open_step % 200 == 0:
                    sim_time = keep_open_step * args_cli.dt
                    wall_time = time.monotonic() - wall_clock_start
                    rtf = sim_time / wall_time if wall_time > 1.0e-6 else 0.0
                    print(
                        f"[TIMING] step={keep_open_step}  sim_time={sim_time:.3f}s  "
                        f"wall_time={wall_time:.3f}s  RTF={rtf:.4f}  "
                        f"(sim is {'faster' if rtf > 1.0 else 'slower'} than real-time by {abs(1.0 - rtf)*100:.1f}%)"
                    )
    except Exception as exc:
        print(f"[ERROR] Environment rebuild runtime failed: {exc}")
        traceback.print_exc()
        raise
    finally:
        if "ros2_bridge" in locals() and ros2_bridge is not None:
            ros2_bridge.shutdown()
        if simulation_app.is_running():
            simulation_app.close()


if __name__ == "__main__":
    main()
