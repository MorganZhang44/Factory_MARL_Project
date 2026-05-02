#!/usr/bin/env python3
"""Validate the first Isaac Sim scene for the simulation module.

This script loads the imported SLAM scene, places two Unitree Go2 robots and one
humanoid intruder, then attaches one camera and one RayCaster LiDAR to each Go2.

It is intentionally standalone inside the Simulation module. It also publishes
the Simulation-Core ROS2 topic contract when launched through
scripts/launch_simulation.sh.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_USD = PROJECT_ROOT / "simulation" / "assets" / "scenes" / "slam_scene.usda"
DEFAULT_EXPORT_USD = PROJECT_ROOT / "simulation" / "assets" / "scenes" / "slam_scene_with_actors.usda"


parser = argparse.ArgumentParser(description="Load SLAM scene and add two Go2 robots, sensors, and one person.")
parser.add_argument("--scene-usd", type=Path, default=DEFAULT_SCENE_USD, help="Pure environment USDA/USD file.")
parser.add_argument("--export-usd", type=Path, default=DEFAULT_EXPORT_USD, help="Where to export the composed scene.")
parser.add_argument("--no-export", action="store_true", help="Do not export a composed USDA file.")
parser.add_argument("--steps", type=int, default=240, help="Number of simulation steps to run for validation.")
parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep.")
parser.add_argument("--keep-open", action="store_true", help="Keep Isaac Sim open after setup for visual inspection.")
parser.add_argument(
    "--view-camera",
    choices=["world", "agent_1", "agent_2"],
    default="world",
    help="Set the active viewport to a robot front camera.",
)
parser.add_argument("--show-lidar", action="store_true", help="Enable RayCaster LiDAR point-cloud debug visualization.")
parser.add_argument("--disable-ros2", action="store_true", help="Do not publish Simulation topics to ROS2.")
parser.add_argument("--topic-prefix", default="/factory/simulation", help="ROS2 topic prefix used by Core.")
parser.add_argument("--control-topic-prefix", default="/factory/control", help="ROS2 control topic prefix used by Core.")
parser.add_argument("--publish-every", type=int, default=4, help="Publish ROS2 data every N simulation steps.")
parser.add_argument("--max-command-age", type=float, default=1.0, help="Ignore stale control commands after this many seconds.")
parser.add_argument("--command-scale", type=float, default=1.0, help="Scale incoming world-frame velocity commands.")
parser.add_argument("--record-video", action="store_true", help="Record a simulation monitor-wall video.")
parser.add_argument("--video-seconds", type=float, default=0.0, help="Record this many simulated seconds when --record-video is enabled.")
parser.add_argument("--video-fps", type=float, default=12.0, help="Output video FPS on the simulation timeline.")
parser.add_argument(
    "--video-output",
    type=Path,
    default=PROJECT_ROOT / "output" / "simulation_legacy_monitor.avi",
    help="Where to save the recorded monitor-wall video.",
)
parser.add_argument(
    "--sensor-view",
    choices=["agent_1", "agent_2"],
    help="Shortcut for live inspection: switch to a dog camera and show its RayCaster LiDAR point cloud.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.sensor_view:
    args_cli.view_camera = args_cli.sensor_view
    args_cli.show_lidar = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import omni.timeline
import omni.usd
import torch
from pxr import Gf, Sdf, UsdGeom

try:
    import isaacsim.core.utils.prims as prim_utils
except ImportError:
    import omni.isaac.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg, RayCaster, RayCasterCfg, patterns
from isaaclab_assets import HUMANOID_CFG
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Image, Imu, LaserScan, PointCloud2, PointField
    from std_msgs.msg import String
except ImportError:
    rclpy = None
    PoseStamped = None
    Image = None
    Imu = None
    LaserScan = None
    PointCloud2 = None
    PointField = None
    String = None


DOGS = {
    "agent_1": {
        "prim_path": "/World/Actors/agent_1",
        "pos": (-2.0, -2.0, 0.42),
        "yaw_deg": 0.0,
    },
    "agent_2": {
        "prim_path": "/World/Actors/agent_2",
        "pos": (-2.0, 1.6, 0.42),
        "yaw_deg": 0.0,
    },
}

STATIC_CUBOIDS = (
    {"name": "Wall_0000_Back", "center": (-0.85, 3.45, 0.9), "size": (9.52, 0.16, 1.8)},
    {"name": "Wall_0001_Left", "center": (-5.5, -0.65, 0.9), "size": (0.16, 8.16, 1.8)},
    {"name": "Wall_0002_Front", "center": (-0.95, -4.65, 0.9), "size": (9.18, 0.16, 1.8)},
    {"name": "Wall_0003_RightLower", "center": (3.75, -2.85, 0.9), "size": (0.16, 4.08, 1.8)},
    {"name": "Wall_0004_StepMiddle", "center": (5.0, -0.95, 0.9), "size": (3.06, 0.16, 1.8)},
    {"name": "Wall_0005_RightUpper", "center": (6.25, 1.35, 0.9), "size": (0.16, 4.93, 1.8)},
    {"name": "Wall_0006_TopRight", "center": (5.15, 3.45, 0.9), "size": (2.89, 0.16, 1.8)},
    {"name": "Wall_0007_StepJoin", "center": (3.75, 2.65, 0.9), "size": (0.16, 2.125, 1.8)},
    {"name": "Obstacle_0000_LeftIsland", "center": (-2.3, -0.85, 0.9), "size": (0.425, 1.275, 1.8)},
    {"name": "Obstacle_0001_CenterIsland", "center": (0.4, -0.95, 0.9), "size": (0.425, 1.375, 1.8)},
    {"name": "Obstacle_0002_RightIsland", "center": (2.95, -0.65, 0.9), "size": (0.45, 1.4, 1.8)},
)
LOCALIZATION_MESH_PATH = "/World/LocalizationStaticMesh"

INTRUDER = {
    "prim_path": "/World/Actors/intruder_1",
    "pos": (2.0, -0.5, 1.34),
    "yaw_deg": 180.0,
}
INTRUDER_ROUTE_ANCHORS = (
    (2.0, -0.5, 1.34),
    (3.8, -0.4, 1.34),
    (3.2, 1.8, 1.34),
    (0.5, 2.3, 1.34),
    (-2.8, 2.0, 1.34),
    (-3.8, 0.1, 1.34),
    (-3.2, -2.6, 1.34),
    (0.2, -3.1, 1.34),
    (2.8, -2.2, 1.34),
    (2.0, -0.5, 1.34),
)
INTRUDER_ROUTE_STEP_SIZE = 0.05
INTRUDER_SPEED_MPS = 0.4

CCTV_PITCH_DEG = 25.0
CCTV_HEIGHT = 2.35
CCTV_CAMERA_NAMES = ("cam_nw", "cam_ne", "cam_e_upper", "cam_e_lower", "cam_se", "cam_sw")
CCTV_CORNER_SPECS = {
    "cam_nw": {"corner": (-5.50, 3.45), "look_hint": (-2.20, 1.90), "mount_inset": 1.05},
    "cam_ne": {"corner": (6.25, 3.45), "look_hint": (2.35, 0.85), "mount_inset": 1.10},
    "cam_e_upper": {"corner": (6.25, -0.95), "look_hint": (2.55, 0.55), "mount_inset": 0.95},
    "cam_e_lower": {"corner": (3.75, -0.95), "look_hint": (2.35, -0.65), "mount_inset": 0.85},
    "cam_se": {"corner": (3.75, -4.65), "look_hint": (2.10, -1.45), "mount_inset": 0.95},
    "cam_sw": {"corner": (-5.50, -4.65), "look_hint": (-2.30, -2.35), "mount_inset": 1.00},
}
DOG_CAMERA_USD_ROT_WXYZ = (0.5, 0.5, -0.5, -0.5)
DOG_CAMERA_POS = (0.3, 0.0, 0.1)
DOG_CAMERA_FOCAL_LENGTH = 3.5
DOG_CAMERA_HORIZONTAL_APERTURE = 12.0
CCTV_CAMERA_FOCAL_LENGTH = 14.0
CCTV_CAMERA_HORIZONTAL_APERTURE = 20.955

PERCEPTION_LIDAR_MOUNT_POS = (0.0, 0.0, 0.35)
PERCEPTION_LIDAR_MAX_DISTANCE = 50.0
PERCEPTION_LIDAR_VERTICAL_FOV_DEG = (-45.0, 45.0)
PERCEPTION_LIDAR_HORIZONTAL_FOV_DEG = (-180.0, 180.0)
DERIVED_SCAN_ANGLE_MIN = -math.pi
DERIVED_SCAN_ANGLE_MAX = math.pi
DERIVED_SCAN_ANGLE_INCREMENT = math.radians(1.0)

class IsaacSimRos2Bridge:
    """Publish Simulation state and receive Core motion commands from ROS2."""

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
        self.node = rclpy.create_node("factory_isaac_sim_publisher")
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
                Image,
                f"{self.topic_prefix}/{robot_id}/camera/semantic_segmentation",
                5,
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
                Image,
                f"{self.topic_prefix}/cctv/{camera_id}/semantic_segmentation",
                5,
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
            robot_id: self.node.create_publisher(
                String,
                f"{self.topic_prefix}/{robot_id}/locomotion/observation",
                20,
            )
            for robot_id in robot_ids
        }
        self.motion_command_sub = self.node.create_subscription(
            String,
            f"{self.control_topic_prefix}/locomotion/motion_command",
            self._on_motion_command,
            10,
        )
        print(f"[INFO] Isaac Sim ROS2 publisher active under {self.topic_prefix}")
        print(f"[INFO] Isaac Sim ROS2 control subscriber active under {self.control_topic_prefix}")

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
        robot_states = {}
        for robot_id in self.robot_ids:
            pos = _root_position(dogs[robot_id])
            robot_states[robot_id] = pos
            self.robot_pose_pubs[robot_id].publish(self._make_pose(robot_id, pos, stamp, _root_quat(dogs[robot_id])))
            self.imu_pubs[robot_id].publish(self._make_imu(robot_id, stamp, dogs[robot_id]))
            self.locomotion_observation_pubs[robot_id].publish(
                String(data=json.dumps(_make_locomotion_observation(robot_id, dogs[robot_id], step_idx)))
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
        self.intruder_pose_pubs["intruder_1"].publish(
            self._make_pose("intruder_1", intruder_pos, stamp, _root_quat(intruder))
        )
        self.state_pub.publish(
            String(
                data=json.dumps(
                    self._make_state(
                        robot_states,
                        intruder_states,
                        step_idx,
                        robot_camera_infos,
                        cctv_camera_infos,
                    )
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

    def _store_motion_command(self, robot_id: str, velocity) -> None:
        if robot_id not in self.robot_ids:
            return
        if not isinstance(velocity, list | tuple) or len(velocity) < 2:
            return
        vx = float(velocity[0])
        vy = float(velocity[1])
        if not math.isfinite(vx) or not math.isfinite(vy):
            return
        self.motion_commands[robot_id] = ([vx, vy], time.monotonic())

    def _store_joint_action(self, robot_id: str, action, action_scale) -> None:
        if robot_id not in self.robot_ids or action is None:
            return
        if not isinstance(action, list | tuple) or len(action) != 12:
            return
        values = [float(value) for value in action]
        if not all(math.isfinite(value) for value in values):
            return
        scale = 0.25 if action_scale is None else float(action_scale)
        self.joint_action_commands[robot_id] = (values, scale, time.monotonic())

    @staticmethod
    def _make_pose(
        entity_id: str,
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
        if quat is None:
            quat = (1.0, 0.0, 0.0, 0.0)
        msg.pose.orientation.w = float(quat[0])
        msg.pose.orientation.x = float(quat[1])
        msg.pose.orientation.y = float(quat[2])
        msg.pose.orientation.z = float(quat[3])
        return msg

    @staticmethod
    def _make_camera_image(robot_id: str, stamp, camera: Camera | None) -> Image | None:
        if camera is None:
            return None
        rgb = camera.data.output.get("rgb")
        if rgb is None or rgb.numel() == 0:
            return None
        image = rgb[0].detach().cpu().numpy()
        if image.ndim != 3 or image.shape[2] < 3:
            return None
        image = image[:, :, :3]
        height, width, _ = image.shape
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = f"{robot_id}/front_camera" if robot_id in DOGS else f"cctv/{robot_id}"
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
        msg.angle_min = DERIVED_SCAN_ANGLE_MIN
        msg.angle_max = DERIVED_SCAN_ANGLE_MAX
        msg.angle_increment = DERIVED_SCAN_ANGLE_INCREMENT
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

    @staticmethod
    def _make_state(
        robot_states: dict[str, tuple[float, float, float]],
        intruder_states: dict[str, tuple[float, float, float]],
        step_idx: int,
        robot_camera_infos: dict[str, dict[str, Any]],
        cctv_camera_infos: dict[str, dict[str, Any]],
    ) -> dict:
        return {
            "timestamp": step_idx,
            "frame_id": "world",
            "robots": {
                robot_id: {"position": list(pos)}
                for robot_id, pos in robot_states.items()
            },
            "intruders": {
                intruder_id: {"position": list(pos)}
                for intruder_id, pos in intruder_states.items()
            },
            "camera_infos": {
                "robots": robot_camera_infos,
                "cctv": cctv_camera_infos,
            },
        }


def _make_quat_z(yaw_deg: float) -> Gf.Quatf:
    """Create a Z-up yaw quaternion for USD xform ops."""
    quat = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), yaw_deg).GetQuat()
    return Gf.Quatf(float(quat.GetReal()), Gf.Vec3f(quat.GetImaginary()))


def _make_quat_xyzw(quat_wxyz: tuple[float, float, float, float]) -> Gf.Quatf:
    """Create a USD quaternion from a (w, x, y, z) tuple."""
    return Gf.Quatf(
        float(quat_wxyz[0]),
        Gf.Vec3f(float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])),
    )


def _quat_multiply(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _normalize_vec(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    if norm < 1.0e-8:
        return (1.0, 0.0, 0.0)
    return (vec[0] / norm, vec[1] / norm, vec[2] / norm)


def _cross_vec(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
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


def _normalize_quat(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quat))
    if norm < 1.0e-8:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(value / norm for value in quat)


def _look_at_with_fixed_pitch_quat(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    pitch_deg: float,
) -> tuple[float, float, float, float]:
    dx = target[0] - eye[0]
    dy = target[1] - eye[1]
    horizontal_norm = math.hypot(dx, dy)
    if horizontal_norm < 1.0e-6:
        forward = (1.0, 0.0, 0.0)
    else:
        pitch_rad = math.radians(pitch_deg)
        forward = (
            math.cos(pitch_rad) * dx / horizontal_norm,
            math.cos(pitch_rad) * dy / horizontal_norm,
            -math.sin(pitch_rad),
        )
    y_axis = _normalize_vec(_cross_vec((0.0, 0.0, 1.0), forward))
    z_axis = _normalize_vec(_cross_vec(forward, y_axis))
    return _quat_from_rotmat(
        (
            (forward[0], y_axis[0], z_axis[0]),
            (forward[1], y_axis[1], z_axis[1]),
            (forward[2], y_axis[2], z_axis[2]),
        )
    )


def _move_toward(source_xy: tuple[float, float], target_xy: tuple[float, float], distance: float) -> tuple[float, float]:
    dx = target_xy[0] - source_xy[0]
    dy = target_xy[1] - source_xy[1]
    norm = math.hypot(dx, dy)
    if norm < 1.0e-6:
        return source_xy
    scale = distance / norm
    return (source_xy[0] + dx * scale, source_xy[1] + dy * scale)


def _cctv_position(camera_id: str) -> tuple[float, float, float]:
    spec = CCTV_CORNER_SPECS[camera_id]
    x, y = _move_toward(spec["corner"], spec["look_hint"], spec["mount_inset"])
    return (x, y, CCTV_HEIGHT)


def _cctv_target(camera_id: str) -> tuple[float, float, float]:
    look_hint = CCTV_CORNER_SPECS[camera_id]["look_hint"]
    return (look_hint[0], look_hint[1], 0.0)


def _set_xform(
    prim_path: str,
    translation: tuple[float, float, float],
    yaw_deg: float = 0.0,
    quat_wxyz: tuple[float, float, float, float] | None = None,
) -> None:
    """Set standard Isaac Lab xform ops on an existing prim."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Cannot set transform. Prim does not exist: {prim_path}")

    if not isinstance(translation, (tuple, list)) or len(translation) != 3:
        raise TypeError(f"Invalid translation for {prim_path}: {translation!r}")
    tx = float(translation[0])
    ty = float(translation[1])
    tz = float(translation[2])
    quat = quat_wxyz if quat_wxyz is not None else _make_quat_z(yaw_deg)
    sim_utils.standardize_xform_ops(
        prim,
        translation=(tx, ty, tz),
        orientation=tuple(float(value) for value in quat),
        scale=(1.0, 1.0, 1.0),
    )


def _create_usd_camera(camera_path: str) -> None:
    """Create a USD pinhole camera under a robot base link."""
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
    position = _cctv_position(camera_id)
    target = _cctv_target(camera_id)
    world_quat = _look_at_with_fixed_pitch_quat(position, target, CCTV_PITCH_DEG)
    usd_quat = _normalize_quat(_quat_multiply(world_quat, DOG_CAMERA_USD_ROT_WXYZ))
    _set_xform(camera_path, position, quat_wxyz=usd_quat)


def _set_active_viewport_camera(camera_path: str) -> None:
    """Switch the main viewport to a USD camera."""
    try:
        from omni.kit.viewport.utility import get_active_viewport
    except ImportError as exc:
        print(f"[WARNING] Cannot import viewport utility: {exc}")
        return

    viewport = get_active_viewport()
    if viewport is None:
        print("[WARNING] No active viewport found. Camera prim exists, but viewport was not switched.")
        return
    viewport.camera_path = camera_path
    print(f"[INFO] Active viewport camera: {camera_path}")


def _create_lidar_readers(lidar_parent_paths: dict[str, str], show_visualization: bool = False) -> dict[str, RayCaster]:
    """Create perception-identical RayCaster LiDAR readers."""
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


def _update_lidar_readers(lidar_readers: dict[str, RayCaster], dt: float) -> None:
    for lidar in lidar_readers.values():
        try:
            lidar.update(dt, force_recompute=True)
        except AttributeError as exc:
            # Some Isaac Lab sensor instances need an explicit reset after sim.reset()
            # before their internal timestamp state exists.
            if "_timestamp" not in str(exc):
                raise
            if hasattr(lidar, "reset"):
                lidar.reset()
            lidar.update(dt, force_recompute=True)


def _print_lidar_status(lidar_readers: dict[str, RayCaster], step_idx: int) -> None:
    """Print a small heartbeat so we know LiDAR frames are being updated."""
    for robot_id, lidar in lidar_readers.items():
        point_cloud = _raycaster_points_sensor_frame(lidar)
        ranges = _derive_planar_scan_from_points(point_cloud)
        point_count = 0 if point_cloud is None else point_cloud.shape[0]
        range_count = 0 if ranges is None else ranges.size
        print(f"[INFO] Step {step_idx}: {robot_id} lidar ranges={range_count} points={point_count}")


def _derive_planar_scan_from_points(points: np.ndarray | None) -> np.ndarray | None:
    if points is None or points.size == 0:
        return None

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    xy_ranges = np.linalg.norm(points[:, :2], axis=1)
    angles = np.arctan2(points[:, 1], points[:, 0])
    valid = (
        np.isfinite(xy_ranges)
        & np.isfinite(angles)
        & (xy_ranges >= 0.05)
        & (xy_ranges <= PERCEPTION_LIDAR_MAX_DISTANCE)
        & (angles >= DERIVED_SCAN_ANGLE_MIN)
        & (angles <= DERIVED_SCAN_ANGLE_MAX)
    )
    if not valid.any():
        return None

    bin_count = int(round((DERIVED_SCAN_ANGLE_MAX - DERIVED_SCAN_ANGLE_MIN) / DERIVED_SCAN_ANGLE_INCREMENT)) + 1
    ranges = np.full((bin_count,), np.inf, dtype=np.float32)
    bin_indices = np.floor((angles[valid] - DERIVED_SCAN_ANGLE_MIN) / DERIVED_SCAN_ANGLE_INCREMENT).astype(np.int32)
    bin_indices = np.clip(bin_indices, 0, bin_count - 1)
    np.minimum.at(ranges, bin_indices, xy_ranges[valid].astype(np.float32))
    return ranges


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


def _create_static_localization_mesh(mesh_path: str = LOCALIZATION_MESH_PATH) -> None:
    """Create the single static mesh used by perception's RayCaster LiDAR."""
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(mesh_path).IsValid():
        stage.RemovePrim(mesh_path)

    points: list[tuple[float, float, float]] = []
    face_vertex_counts: list[int] = []
    face_vertex_indices: list[int] = []

    def add_triangle(i0: int, i1: int, i2: int) -> None:
        face_vertex_counts.append(3)
        face_vertex_indices.extend([i0, i1, i2])

    def add_box(center: tuple[float, float, float], size: tuple[float, float, float]) -> None:
        cx, cy, cz = center
        sx, sy, sz = size
        hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
        base_index = len(points)
        points.extend(
            [
                (cx - hx, cy - hy, cz - hz),
                (cx + hx, cy - hy, cz - hz),
                (cx + hx, cy + hy, cz - hz),
                (cx - hx, cy + hy, cz - hz),
                (cx - hx, cy - hy, cz + hz),
                (cx + hx, cy - hy, cz + hz),
                (cx + hx, cy + hy, cz + hz),
                (cx - hx, cy + hy, cz + hz),
            ]
        )
        for tri in (
            (0, 1, 2), (0, 2, 3),
            (4, 6, 5), (4, 7, 6),
            (0, 4, 5), (0, 5, 1),
            (1, 5, 6), (1, 6, 2),
            (2, 6, 7), (2, 7, 3),
            (3, 7, 4), (3, 4, 0),
        ):
            add_triangle(base_index + tri[0], base_index + tri[1], base_index + tri[2])

    for cuboid in STATIC_CUBOIDS:
        add_box(cuboid["center"], cuboid["size"])

    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    UsdGeom.Imageable(mesh).MakeInvisible()
    print(f"[INFO] Localization mesh: {mesh_path} ({len(points)} vertices, {len(face_vertex_counts)} triangles)")


def _load_environment(scene_usd: Path) -> None:
    """Reference the pure scene under /World/SlamScene."""
    if not scene_usd.exists():
        raise FileNotFoundError(f"Scene file does not exist: {scene_usd}")

    prim_utils.create_prim("/World", "Xform")
    prim_utils.create_prim("/World/SlamScene", "Xform", usd_path=str(scene_usd))
    prim_utils.create_prim("/World/Actors", "Xform")


def _spawn_actors() -> tuple[dict[str, Articulation], Articulation]:
    """Spawn two Go2 robots and one humanoid intruder."""
    dogs: dict[str, Articulation] = {}
    for dog_id, spec in DOGS.items():
        cfg = UNITREE_GO2_CFG.replace(prim_path=spec["prim_path"])
        cfg.spawn.semantic_tags = [("class", "dog")]
        dogs[dog_id] = Articulation(cfg)

    intruder_cfg = HUMANOID_CFG.replace(prim_path=INTRUDER["prim_path"])
    intruder_cfg.spawn.semantic_tags = [("class", "suspect")]
    intruder = Articulation(intruder_cfg)
    return dogs, intruder


def _attach_sensors(dog_specs: dict[str, dict]) -> tuple[list[str], dict[str, str]]:
    """Attach camera prims and return LiDAR parent bodies for each dog."""
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
    """Create Isaac Lab camera readers for the dog front cameras."""
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


def _camera_rgb_frame(camera: Camera | None) -> np.ndarray | None:
    if camera is None:
        return None
    rgb = camera.data.output.get("rgb")
    if rgb is None or rgb.numel() == 0:
        return None
    image = rgb[0].detach().cpu().numpy()
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        return None
    return image


def _render_video_wall(
    dog_readers: dict[str, Camera],
    cctv_readers: dict[str, Camera],
    feed_order: list[str],
    tile_size: tuple[int, int] = (480, 360),
) -> np.ndarray | None:
    if cv2 is None:
        return None

    frames: list[np.ndarray] = []
    prototype_frame: np.ndarray | None = None
    rendered_frames: dict[str, np.ndarray] = {}

    merged_readers: dict[str, Camera] = {**dog_readers, **cctv_readers}
    for feed_id in feed_order:
        frame = _camera_rgb_frame(merged_readers.get(feed_id))
        if frame is None:
            continue
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, feed_id.upper(), (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        rendered_frames[feed_id] = bgr
        if prototype_frame is None:
            prototype_frame = bgr

    if prototype_frame is None:
        return None

    for feed_id in feed_order:
        if feed_id not in rendered_frames:
            placeholder = np.zeros_like(prototype_frame)
            cv2.putText(placeholder, feed_id.upper(), (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            rendered_frames[feed_id] = placeholder
        frame = rendered_frames[feed_id]
        interpolation = cv2.INTER_AREA if frame.shape[1] >= tile_size[0] else cv2.INTER_LINEAR
        frames.append(cv2.resize(frame, tile_size, interpolation=interpolation))

    cols = 3 if len(frames) > 4 else 2 if len(frames) > 1 else 1
    rows = math.ceil(len(frames) / cols)
    while len(frames) < rows * cols:
        frames.append(np.zeros_like(frames[0]))
    row_images = []
    for row_idx in range(rows):
        start = row_idx * cols
        row_images.append(np.hstack(frames[start:start + cols]))
    return np.vstack(row_images)


class SimulationVideoRecorder:
    def __init__(self, output_path: Path, fps: float, feed_order: list[str]) -> None:
        if cv2 is None:
            raise RuntimeError("cv2 is unavailable; cannot record simulation video.")
        self.output_path = output_path
        self.fps = float(max(fps, 1.0))
        self.feed_order = feed_order
        self._writer = None

    def maybe_write(self, dog_readers: dict[str, Camera], cctv_readers: dict[str, Camera]) -> None:
        wall = _render_video_wall(dog_readers, cctv_readers, self.feed_order)
        if wall is None:
            return
        if self._writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self._writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                (wall.shape[1], wall.shape[0]),
            )
            print(f"[INFO] Started recording simulation video to {self.output_path}")
        self._writer.write(wall)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            print(f"[INFO] Simulation video saved to {self.output_path}")


def _update_camera_readers(camera_readers: dict[str, Camera], dt: float) -> None:
    for camera in camera_readers.values():
        camera.update(dt)


def _reset_sensor_readers(camera_readers: dict[str, Camera], lidar_readers: dict[str, RayCaster]) -> None:
    for camera in camera_readers.values():
        if hasattr(camera, "reset"):
            camera.reset()
    for lidar in lidar_readers.values():
        if hasattr(lidar, "reset"):
            lidar.reset()


def _write_default_state(entity: Articulation, pos: tuple[float, float, float]) -> None:
    """Place an articulation root at a world position and write default joints."""
    root_state = entity.data.default_root_state.clone()
    root_state[:, :3] = torch.tensor(pos, device=root_state.device, dtype=root_state.dtype)
    entity.write_root_state_to_sim(root_state)
    entity.write_joint_state_to_sim(entity.data.default_joint_pos.clone(), entity.data.default_joint_vel.clone())
    entity.reset()


def _interpolate_route(
    anchors: tuple[tuple[float, float, float], ...],
    step_size: float,
) -> list[tuple[float, float, float]]:
    route: list[tuple[float, float, float]] = []
    if len(anchors) < 2:
        return list(anchors)
    for idx in range(len(anchors) - 1):
        p0 = anchors[idx]
        p1 = anchors[idx + 1]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dz = p1[2] - p0[2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        steps = max(1, int(distance / max(step_size, 1.0e-4)))
        for step in range(steps):
            alpha = step / steps
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
INTRUDER_ARM_DOWN_POSE = {
    "left_upper_arm:0": 0.9,
    "left_upper_arm:2": -0.8,
    "right_upper_arm:0": 0.9,
    "right_upper_arm:2": -0.8,
    "left_lower_arm": -2,
    "right_lower_arm": -2,
}



def _initialize_actor_states(dogs: dict[str, Articulation], intruder: Articulation) -> None:
    for dog_id, dog in dogs.items():
        _write_default_state(dog, DOGS[dog_id]["pos"])
    _write_default_state(intruder, INTRUDER["pos"])


def _root_position(entity: Articulation) -> tuple[float, float, float]:
    pos = entity.data.root_pos_w[0].detach().cpu().tolist()
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def _root_quat(entity: Articulation) -> tuple[float, float, float, float]:
    quat = entity.data.root_quat_w[0].detach().cpu().tolist()
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def _tensor_1d(entity: Articulation, name: str, size: int, default: float = 0.0) -> list[float]:
    value = getattr(entity.data, name, None)
    if value is None:
        return [default] * size
    data = value[0].detach().cpu().tolist()
    return [float(item) for item in data[:size]]


def _make_locomotion_observation(robot_id: str, dog: Articulation, step_idx: int) -> dict:
    """Build the 48D observation used by the Unitree Go2 flat velocity policy."""
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
    return {
        "robot_id": robot_id,
        "timestamp": step_idx,
        "observation": observation,
        "schema": "go2_flat_velocity_policy_obs_v1",
    }


def _move_intruder_along_route(intruder: Articulation, step_idx: int, dt: float) -> None:
    """Move the intruder along a fixed closed route while keeping it upright."""
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
    """Cache the current humanoid joint pose so later route motion can reuse it."""
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
    """Force the humanoid back to its cached standing joint pose every step."""
    joint_pos = getattr(intruder, "_factory_locked_joint_pos", None)
    joint_vel = getattr(intruder, "_factory_locked_joint_vel", None)
    if joint_pos is None or joint_vel is None:
        joint_pos = intruder.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(intruder.data.default_joint_vel)
    intruder.write_joint_state_to_sim(joint_pos, joint_vel)
    intruder.set_joint_position_target(joint_pos)


def _apply_motion_commands(
    dogs: dict[str, Articulation],
    commands: dict[str, list[float]],
    joint_actions: dict[str, tuple[list[float], float]],
    dt: float,
    command_scale: float,
) -> set[str]:
    """Apply Core commands.

    Low-level joint actions take priority. World-frame root velocity is kept as
    a fallback for debugging when the policy service is not ready yet.
    """
    low_level_applied: set[str] = set()
    for robot_id, (action, action_scale) in joint_actions.items():
        dog = dogs.get(robot_id)
        if dog is None:
            continue
        action_tensor = torch.tensor(action, device=dog.data.joint_pos.device, dtype=dog.data.joint_pos.dtype).unsqueeze(0)
        target = dog.data.default_joint_pos + action_tensor * float(action_scale)
        dog.set_joint_position_target(target)
        dog._factory_last_low_level_action = action
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


def _validate_prims(paths: list[str]) -> None:
    stage = omni.usd.get_context().get_stage()
    missing = [path for path in paths if not stage.GetPrimAtPath(path).IsValid()]
    if missing:
        raise RuntimeError(f"Missing expected prims: {missing}")


def _export_stage(export_usd: Path) -> None:
    export_usd.parent.mkdir(parents=True, exist_ok=True)
    stage = omni.usd.get_context().get_stage()
    if not stage.GetRootLayer().Export(str(export_usd)):
        raise RuntimeError(f"Failed to export stage to {export_usd}")
    print(f"[INFO] Exported composed scene: {export_usd}")


def main() -> None:
    print(f"[INFO] Loading pure scene: {args_cli.scene_usd}")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args_cli.dt, device=args_cli.device))
    sim.set_camera_view(eye=[6.0, -7.0, 5.0], target=[0.0, -0.5, 0.5])

    _load_environment(args_cli.scene_usd)
    _create_static_localization_mesh()
    dogs, intruder = _spawn_actors()
    sensor_paths, lidar_paths = _attach_sensors(DOGS)
    cctv_paths = _attach_cctv_sensors()
    camera_readers = _create_camera_readers(DOGS)
    cctv_readers = _create_cctv_readers()
    lidar_readers = _create_lidar_readers(lidar_paths, show_visualization=args_cli.show_lidar)
    recorder = None
    if args_cli.record_video:
        feed_order = [*DOGS.keys(), *CCTV_CAMERA_NAMES]
        recorder = SimulationVideoRecorder(
            output_path=args_cli.video_output,
            fps=args_cli.video_fps,
            feed_order=feed_order,
        )

    expected_prims = [
        "/World/SlamScene",
        "/World/Actors",
        "/World/CCTV",
        INTRUDER["prim_path"],
        *[spec["prim_path"] for spec in DOGS.values()],
        *sensor_paths,
        *cctv_paths,
    ]
    _validate_prims(expected_prims)

    sim.reset()
    _reset_sensor_readers(camera_readers, {})
    _reset_sensor_readers(cctv_readers, {})
    _initialize_actor_states(dogs, intruder)
    sim.step()
    for dog in dogs.values():
        dog.update(args_cli.dt)
    intruder.update(args_cli.dt)
    _capture_intruder_locked_pose(intruder)
    ros2_bridge = None if args_cli.disable_ros2 else IsaacSimRos2Bridge(
        topic_prefix=args_cli.topic_prefix,
        control_topic_prefix=args_cli.control_topic_prefix,
        robot_ids=list(DOGS.keys()),
        intruder_ids=["intruder_1"],
        max_command_age=args_cli.max_command_age,
    )
    if args_cli.view_camera != "world":
        _set_active_viewport_camera(f"{DOGS[args_cli.view_camera]['prim_path']}/base/front_camera")
    omni.timeline.get_timeline_interface().play()
    print("[INFO] Scene setup complete. Running validation steps...")
    _wall_clock_start = time.monotonic()
    effective_steps = args_cli.steps
    if args_cli.record_video and args_cli.video_seconds > 0.0:
        effective_steps = max(1, int(math.ceil(args_cli.video_seconds / max(args_cli.dt, 1.0e-6))))
        print(
            f"[INFO] Recording {args_cli.video_seconds:.3f} simulated seconds at dt={args_cli.dt:.6f}, "
            f"steps={effective_steps}, video_fps={args_cli.video_fps:.2f}"
        )
    capture_interval_sim = 1.0 / max(args_cli.video_fps, 1.0)
    next_capture_sim_time = 0.0

    for step_idx in range(effective_steps):
        low_level_applied: set[str] = set()
        if ros2_bridge is not None:
            ros2_bridge.spin_once()
            low_level_applied = _apply_motion_commands(
                dogs,
                ros2_bridge.current_motion_commands(),
                ros2_bridge.current_joint_actions(),
                args_cli.dt,
                args_cli.command_scale,
            )

        for dog_id, dog in dogs.items():
            if dog_id in low_level_applied:
                dog.write_data_to_sim()
                continue
            dog.set_joint_position_target(dog.data.default_joint_pos)
            dog.write_data_to_sim()
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
        if recorder is not None:
            current_sim_time = (step_idx + 1) * args_cli.dt
            while current_sim_time + 1.0e-9 >= next_capture_sim_time:
                recorder.maybe_write(camera_readers, cctv_readers)
                next_capture_sim_time += capture_interval_sim
        if ros2_bridge is not None and step_idx % max(1, args_cli.publish_every) == 0:
            ros2_bridge.publish(dogs, intruder, camera_readers, cctv_readers, lidar_readers, step_idx)

        if step_idx in (0, effective_steps - 1):
            positions = {
                dog_id: dog.data.root_pos_w[0].detach().cpu().tolist()
                for dog_id, dog in dogs.items()
            }
            positions["intruder_1"] = intruder.data.root_pos_w[0].detach().cpu().tolist()
            print(f"[INFO] Step {step_idx + 1}/{effective_steps} actor positions: {positions}")
        if lidar_readers and step_idx in (0, effective_steps - 1):
            _print_lidar_status(lidar_readers, step_idx + 1)
        if step_idx > 0 and step_idx % 200 == 0:
            _sim_time = (step_idx + 1) * args_cli.dt
            _wall_time = time.monotonic() - _wall_clock_start
            _rtf = _sim_time / _wall_time if _wall_time > 1e-6 else 0.0
            print(
                f"[TIMING] step={step_idx}  sim_time={_sim_time:.3f}s  "
                f"wall_time={_wall_time:.3f}s  RTF={_rtf:.4f}  "
                f"(sim is {'faster' if _rtf > 1.0 else 'slower'} than real-time by {abs(1.0 - _rtf)*100:.1f}%)"
            )

    if not args_cli.no_export:
        _export_stage(args_cli.export_usd)

    if recorder is not None:
        recorder.close()

    if args_cli.keep_open:
        print("[INFO] Keeping Isaac Sim open. Close the Isaac Sim window to stop.")
        keep_open_step = args_cli.steps
        while simulation_app.is_running():
            low_level_applied = set()
            if ros2_bridge is not None:
                ros2_bridge.spin_once()
                low_level_applied = _apply_motion_commands(
                    dogs,
                    ros2_bridge.current_motion_commands(),
                    ros2_bridge.current_joint_actions(),
                    args_cli.dt,
                    args_cli.command_scale,
                )

            for dog_id, dog in dogs.items():
                if dog_id in low_level_applied:
                    dog.write_data_to_sim()
                    continue
                dog.set_joint_position_target(dog.data.default_joint_pos)
                dog.write_data_to_sim()
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
            if keep_open_step % 200 == 0:
                _sim_time = keep_open_step * args_cli.dt
                _wall_time = time.monotonic() - _wall_clock_start
                _rtf = _sim_time / _wall_time if _wall_time > 1e-6 else 0.0
                print(
                    f"[TIMING] step={keep_open_step}  sim_time={_sim_time:.3f}s  "
                    f"wall_time={_wall_time:.3f}s  RTF={_rtf:.4f}  "
                    f"(sim is {'faster' if _rtf > 1.0 else 'slower'} than real-time by {abs(1.0 - _rtf)*100:.1f}%)"
                )

    if ros2_bridge is not None:
        ros2_bridge.shutdown()

    print("[INFO] Validation complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        carb.log_error(f"Simulation validation failed: {exc}")
        raise
    finally:
        simulation_app.close()
