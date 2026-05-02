#!/usr/bin/env python3
"""Fresh simulation runtime built from the migrated environment_updated package.

This entrypoint is intentionally isolated from the existing USDA-based runtime.
It uses the migrated cfg-driven environment package and adapts it to the
current project's ROS2 topic contract.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher
try:
    import cv2
except ImportError:
    cv2 = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIMULATION_ROOT = PROJECT_ROOT / "simulation"
ENV_ROOT = SIMULATION_ROOT / "environment_rewrite"
DEFAULT_SCENE_USD = ENV_ROOT / "assets" / "scenes" / "slam_scene.usda"

if str(SIMULATION_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATION_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


parser = argparse.ArgumentParser(description="Run the migrated environment_updated scene with the current ROS2 contract.")
parser.add_argument("--scene-usd", type=Path, default=DEFAULT_SCENE_USD, help="Static USDA/USD scene file for the rewrite line.")
parser.add_argument("--steps", type=int, default=240, help="Number of simulation steps to run.")
parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep.")
parser.add_argument("--keep-open", action="store_true", help="Keep Isaac Sim open after setup.")
parser.add_argument(
    "--view-camera",
    choices=["world", "agent_1", "agent_2"],
    default="world",
    help="Set the active viewport to a robot front camera.",
)
parser.add_argument("--show-lidar", action="store_true", help="Enable RayCaster LiDAR debug visualization.")
parser.add_argument("--disable-ros2", action="store_true", help="Do not publish simulation topics to ROS2.")
parser.add_argument("--topic-prefix", default="/factory/simulation", help="ROS2 topic prefix used by Core.")
parser.add_argument("--control-topic-prefix", default="/factory/control", help="ROS2 control topic prefix used by Core.")
parser.add_argument("--publish-every", type=int, default=4, help="Publish ROS2 data every N simulation steps.")
parser.add_argument("--max-command-age", type=float, default=1.0, help="Ignore stale control commands older than this.")
parser.add_argument("--command-scale", type=float, default=1.0, help="Scale incoming world-frame velocity commands.")
parser.add_argument("--record-video", action="store_true", help="Record a simulation monitor-wall video.")
parser.add_argument("--video-seconds", type=float, default=0.0, help="Record this many simulated seconds when --record-video is enabled.")
parser.add_argument("--video-fps", type=float, default=12.0, help="Output video FPS on the simulation timeline.")
parser.add_argument(
    "--video-output",
    type=Path,
    default=PROJECT_ROOT / "output" / "simulation_rewrite_monitor.avi",
    help="Where to save the recorded monitor-wall video.",
)
parser.add_argument(
    "--sensor-view",
    choices=["agent_1", "agent_2"],
    help="Shortcut for live inspection: switch to a dog camera and show LiDAR debug view.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.sensor_view:
    args_cli.view_camera = args_cli.sensor_view
    args_cli.show_lidar = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import Camera, CameraCfg, Imu as ImuSensor, RayCaster

from environment_rewrite.config.scene_cfg import SurveillanceSceneCfg  # noqa: E402
from environment_rewrite.localization_mesh import create_static_localization_mesh  # noqa: E402
from environment_rewrite.static_scene_geometry import MAP_BOUNDS, get_actor_spawn_points, get_surveillance_camera_names  # noqa: E402

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


ROBOT_NAME_MAP = {
    "agent_1": "go2_dog_1",
    "agent_2": "go2_dog_2",
}
ROBOT_CAMERA_MAP = {
    "agent_1": "dog1_cam",
    "agent_2": "dog2_cam",
}
ROBOT_LIDAR_MAP = {
    "agent_1": "dog1_lidar",
    "agent_2": "dog2_lidar",
}
ROBOT_IMU_MAP = {
    "agent_1": "go2_dog_1_imu",
    "agent_2": "go2_dog_2_imu",
}
INTRUDER_SCENE_NAME = "suspect"
INTRUDER_TOPIC_NAME = "intruder_1"
CCTV_NAMES = tuple(get_surveillance_camera_names())
SPAWN_POINTS = get_actor_spawn_points()
DOG_DEFAULT_YAW_DEG = 0.0
SUSPECT_DEFAULT_YAW_DEG = 180.0

PERCEPTION_LIDAR_MAX_DISTANCE = 50.0
DERIVED_SCAN_ANGLE_MIN = -math.pi
DERIVED_SCAN_ANGLE_MAX = math.pi
DERIVED_SCAN_ANGLE_INCREMENT = math.radians(1.0)
DOG_CAMERA_USD_ROT_WXYZ = (0.5, 0.5, -0.5, -0.5)
DOG_CAMERA_POS = (0.3, 0.0, 0.1)
DOG_CAMERA_FOCAL_LENGTH = 3.5
DOG_CAMERA_HORIZONTAL_APERTURE = 12.0


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
    dog_cameras: dict[str, Camera],
    cctv_cameras: dict[str, Camera],
    feed_order: list[str],
    tile_size: tuple[int, int] = (480, 360),
) -> np.ndarray | None:
    if cv2 is None:
        return None
    frames: list[np.ndarray] = []
    prototype_frame: np.ndarray | None = None
    rendered_frames: dict[str, np.ndarray] = {}
    merged_readers: dict[str, Camera] = {**dog_cameras, **cctv_cameras}
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

    def maybe_write(self, dog_cameras: dict[str, Camera], cctv_cameras: dict[str, Camera]) -> None:
        wall = _render_video_wall(dog_cameras, cctv_cameras, self.feed_order)
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


class IsaacSimRos2Bridge:
    """Publish simulation state and receive Core motion commands."""

    def __init__(
        self,
        topic_prefix: str,
        control_topic_prefix: str,
        robot_ids: list[str],
        intruder_ids: list[str],
        max_command_age: float,
    ) -> None:
        if rclpy is None:
            raise RuntimeError("rclpy is unavailable. Launch through the simulation launcher with Isaac ROS2 paths configured.")

        rclpy.init(args=None)
        self.node = rclpy.create_node("factory_environment_rewrite_sim")
        self.topic_prefix = topic_prefix.rstrip("/")
        self.control_topic_prefix = control_topic_prefix.rstrip("/")
        self.robot_ids = robot_ids
        self.intruder_ids = intruder_ids
        self.cctv_ids = list(CCTV_NAMES)
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
            robot_id: self.node.create_publisher(Image, f"{self.topic_prefix}/{robot_id}/camera/semantic_segmentation", 5)
            for robot_id in robot_ids
        }
        self.imu_pubs = {
            robot_id: self.node.create_publisher(Imu, f"{self.topic_prefix}/{robot_id}/imu", 20)
            for robot_id in robot_ids
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
        self.cctv_camera_pubs = {
            camera_id: self.node.create_publisher(Image, f"{self.topic_prefix}/cctv/{camera_id}/image_raw", 5)
            for camera_id in self.cctv_ids
        }
        self.cctv_semantic_pubs = {
            camera_id: self.node.create_publisher(Image, f"{self.topic_prefix}/cctv/{camera_id}/semantic_segmentation", 5)
            for camera_id in self.cctv_ids
        }
        self.motion_command_sub = self.node.create_subscription(
            String,
            f"{self.control_topic_prefix}/locomotion/motion_command",
            self._on_motion_command,
            10,
        )
        print(f"[INFO] Rewrite simulation ROS2 publisher active under {self.topic_prefix}")

    def publish(
        self,
        dogs: dict[str, Articulation],
        intruder: Articulation,
        dog_cameras: dict[str, Camera],
        cctv_cameras: dict[str, Camera],
        dog_lidars: dict[str, RayCaster],
        dog_imus: dict[str, ImuSensor],
        step_idx: int,
    ) -> None:
        stamp = self.node.get_clock().now().to_msg()
        robot_states = {}
        for robot_id, dog in dogs.items():
            pos = _root_position(dog)
            robot_states[robot_id] = pos
            self.robot_pose_pubs[robot_id].publish(self._make_pose(pos, stamp, _root_quat(dog)))
            self.imu_pubs[robot_id].publish(self._make_imu(robot_id, stamp, dog_imus[robot_id]))
            self.locomotion_observation_pubs[robot_id].publish(
                String(data=json.dumps(_make_locomotion_observation(robot_id, dog, step_idx)))
            )

            camera = dog_cameras.get(robot_id)
            camera_msg = self._make_camera_image(robot_id, stamp, camera)
            if camera_msg is not None:
                self.camera_pubs[robot_id].publish(camera_msg)
            depth_msg = self._make_depth_image(robot_id, stamp, camera)
            if depth_msg is not None:
                self.depth_pubs[robot_id].publish(depth_msg)
            semantic_msg = self._make_semantic_image(robot_id, stamp, camera)
            if semantic_msg is not None:
                self.semantic_pubs[robot_id].publish(semantic_msg)

            lidar = dog_lidars.get(robot_id)
            scan_msg = self._make_scan(robot_id, stamp, lidar)
            if scan_msg is not None:
                self.lidar_pubs[robot_id].publish(scan_msg)
            points_msg = self._make_point_cloud(robot_id, stamp, lidar)
            if points_msg is not None:
                self.lidar_point_cloud_pubs[robot_id].publish(points_msg)

        for camera_id in self.cctv_ids:
            camera = cctv_cameras.get(camera_id)
            camera_msg = self._make_camera_image(camera_id, stamp, camera)
            if camera_msg is not None:
                self.cctv_camera_pubs[camera_id].publish(camera_msg)
            semantic_msg = self._make_semantic_image(camera_id, stamp, camera)
            if semantic_msg is not None:
                self.cctv_semantic_pubs[camera_id].publish(semantic_msg)

        intruder_pos = _root_position(intruder)
        intruder_states = {INTRUDER_TOPIC_NAME: intruder_pos}
        self.intruder_pose_pubs[INTRUDER_TOPIC_NAME].publish(self._make_pose(intruder_pos, stamp, _root_quat(intruder)))
        self.state_pub.publish(String(data=json.dumps(self._make_state(robot_states, intruder_states, step_idx))))

    def spin_once(self) -> None:
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def current_motion_commands(self) -> dict[str, list[float]]:
        now = time.monotonic()
        return {
            robot_id: velocity
            for robot_id, (velocity, updated_at) in self.motion_commands.items()
            if now - updated_at <= self.max_command_age
        }

    def current_joint_actions(self) -> dict[str, tuple[list[float], float]]:
        now = time.monotonic()
        return {
            robot_id: (action, scale)
            for robot_id, (action, scale, updated_at) in self.joint_action_commands.items()
            if now - updated_at <= self.max_command_age
        }

    def shutdown(self) -> None:
        self.node.destroy_node()
        rclpy.shutdown()

    def _on_motion_command(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            payload = data.get("payload", data)
            commands = payload.get("commands")
            if isinstance(commands, dict):
                for robot_id, command in commands.items():
                    self._store_motion_command(str(robot_id), command.get("velocity"))
                    self._store_joint_action(str(robot_id), command.get("action"), command.get("action_scale"))
                return
            robot_id = payload.get("robot_id")
            if robot_id is not None:
                self._store_motion_command(str(robot_id), payload.get("velocity"))
                self._store_joint_action(str(robot_id), payload.get("action"), payload.get("action_scale"))
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"[WARNING] Invalid motion command JSON: {exc}")

    def _store_motion_command(self, robot_id: str, velocity) -> None:
        if robot_id not in self.robot_ids:
            return
        if not isinstance(velocity, (list, tuple)) or len(velocity) < 2:
            return
        vx = float(velocity[0])
        vy = float(velocity[1])
        if not math.isfinite(vx) or not math.isfinite(vy):
            return
        self.motion_commands[robot_id] = ([vx, vy], time.monotonic())

    def _store_joint_action(self, robot_id: str, action, action_scale) -> None:
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
    def _make_pose(pos: tuple[float, float, float], stamp, quat: tuple[float, float, float, float]) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "world"
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
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
        height, width, _ = image.shape
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = f"{sensor_id}/front_camera" if sensor_id in ROBOT_NAME_MAP else f"cctv/{sensor_id}"
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
        depth = camera.data.output.get("depth")
        if depth is None or depth.numel() == 0:
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
        image = semantic[0].detach().cpu().numpy()
        if image.ndim == 3:
            image = image[:, :, 0] if image.shape[-1] == 1 else image[0]
        if image.ndim != 2:
            return None
        image = image.astype("<i4", copy=False)
        height, width = image.shape
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = f"{sensor_id}/front_camera" if sensor_id in ROBOT_NAME_MAP else f"cctv/{sensor_id}"
        msg.height = height
        msg.width = width
        msg.encoding = "32SC1"
        msg.is_bigendian = False
        msg.step = width * 4
        msg.data = image.tobytes()
        return msg

    @staticmethod
    def _make_imu(robot_id: str, stamp, imu_sensor: ImuSensor) -> Imu:
        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = f"{robot_id}/base"
        quat = imu_sensor.data.quat_w[0].detach().cpu().tolist()
        ang_vel = imu_sensor.data.ang_vel_b[0].detach().cpu().tolist()
        lin_acc = imu_sensor.data.lin_acc_b[0].detach().cpu().tolist()
        msg.orientation.w = float(quat[0])
        msg.orientation.x = float(quat[1])
        msg.orientation.y = float(quat[2])
        msg.orientation.z = float(quat[3])
        msg.angular_velocity.x = float(ang_vel[0])
        msg.angular_velocity.y = float(ang_vel[1])
        msg.angular_velocity.z = float(ang_vel[2])
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
        msg.intensities = [1.0 if math.isfinite(float(v)) else 0.0 for v in ranges]
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
    def _make_state(
        robot_states: dict[str, tuple[float, float, float]],
        intruder_states: dict[str, tuple[float, float, float]],
        step_idx: int,
    ) -> dict:
        return {
            "timestamp": step_idx,
            "frame_id": "world",
            "robots": {robot_id: {"position": list(pos)} for robot_id, pos in robot_states.items()},
            "intruders": {intruder_id: {"position": list(pos)} for intruder_id, pos in intruder_states.items()},
        }


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
        + [float(v) for v in joint_pos_rel[:12]]
        + [float(v) for v in joint_vel_rel[:12]]
        + [float(v) for v in last_action[:12]]
    )
    return {
        "robot_id": robot_id,
        "timestamp": step_idx,
        "observation": observation,
        "schema": "go2_flat_velocity_policy_obs_v1",
    }


def _quat_wxyz_from_yaw_deg(yaw_deg: float) -> tuple[float, float, float, float]:
    half = math.radians(yaw_deg) * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _initialize_actor_baselines(dogs: dict[str, Articulation], suspect: Articulation) -> None:
    dog_quat = _quat_wxyz_from_yaw_deg(DOG_DEFAULT_YAW_DEG)
    for robot_id, dog in dogs.items():
        spawn_key = "dog1" if robot_id == "agent_1" else "dog2"
        root_state = dog.data.root_state_w.clone()
        root_state[:, 0] = SPAWN_POINTS[spawn_key][0]
        root_state[:, 1] = SPAWN_POINTS[spawn_key][1]
        root_state[:, 2] = SPAWN_POINTS[spawn_key][2]
        root_state[:, 3] = dog_quat[0]
        root_state[:, 4] = dog_quat[1]
        root_state[:, 5] = dog_quat[2]
        root_state[:, 6] = dog_quat[3]
        root_state[:, 7:13] = 0.0
        dog.write_root_state_to_sim(root_state)
        dog.write_joint_state_to_sim(dog.data.default_joint_pos.clone(), dog.data.default_joint_vel.clone())
        dog._factory_last_low_level_action = [0.0] * 12

    suspect_root = suspect.data.root_state_w.clone()
    suspect_quat = _quat_wxyz_from_yaw_deg(SUSPECT_DEFAULT_YAW_DEG)
    suspect_root[:, 0] = SPAWN_POINTS["suspect"][0]
    suspect_root[:, 1] = SPAWN_POINTS["suspect"][1]
    suspect_root[:, 2] = SPAWN_POINTS["suspect"][2]
    suspect_root[:, 3] = suspect_quat[0]
    suspect_root[:, 4] = suspect_quat[1]
    suspect_root[:, 5] = suspect_quat[2]
    suspect_root[:, 6] = suspect_quat[3]
    suspect_root[:, 7:13] = 0.0
    suspect.write_root_state_to_sim(suspect_root)
    suspect.write_joint_state_to_sim(suspect.data.default_joint_pos.clone(), suspect.data.default_joint_vel.clone())


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


def _stabilize_suspect(suspect: Articulation) -> None:
    root_state = suspect.data.root_state_w.clone()
    yaw_quat = _quat_wxyz_from_yaw_deg(SUSPECT_DEFAULT_YAW_DEG)
    root_state[:, 0] = SPAWN_POINTS["suspect"][0]
    root_state[:, 1] = SPAWN_POINTS["suspect"][1]
    root_state[:, 2] = SPAWN_POINTS["suspect"][2]
    root_state[:, 3] = yaw_quat[0]
    root_state[:, 4] = yaw_quat[1]
    root_state[:, 5] = yaw_quat[2]
    root_state[:, 6] = yaw_quat[3]
    root_state[:, 7:13] = 0.0
    suspect.write_root_state_to_sim(root_state)


def _raycaster_points_sensor_frame(lidar: RayCaster) -> np.ndarray | None:
    hits_w = lidar.data.ray_hits_w
    pos_w = lidar.data.pos_w
    quat_w = lidar.data.quat_w
    if hits_w is None or pos_w is None or quat_w is None or hits_w.numel() == 0:
        return None
    hits = hits_w[0].detach().cpu().numpy().astype(np.float32, copy=False)
    origin = pos_w[0].detach().cpu().numpy().astype(np.float32, copy=False)
    quat = quat_w[0].detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(origin).all() or not np.isfinite(quat).all():
        return None
    valid_hits = np.isfinite(hits).all(axis=1)
    hits = hits[valid_hits]
    if hits.size == 0:
        return None
    points = _rotate_points_by_inverse_quat(hits - origin, quat)
    ranges = np.linalg.norm(points, axis=1)
    valid = np.isfinite(points).all(axis=1) & (ranges <= PERCEPTION_LIDAR_MAX_DISTANCE)
    points = points[valid]
    return points.astype(np.float32) if points.size else None


def _rotate_points_by_inverse_quat(points: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat_wxyz))
    if not math.isfinite(norm) or norm < 1.0e-8:
        return np.empty((0, 3), dtype=np.float32)
    w, x, y, z = [float(v / norm) for v in quat_wxyz]
    rot = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return points @ rot


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


def _set_active_viewport_camera(camera_path: str) -> None:
    try:
        from omni.kit.viewport.utility import get_active_viewport
    except ImportError as exc:
        print(f"[WARNING] Cannot import viewport utility: {exc}")
        return
    viewport = get_active_viewport()
    if viewport is None:
        print("[WARNING] No active viewport found.")
        return
    viewport.camera_path = camera_path
    print(f"[INFO] Active viewport camera: {camera_path}")


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


def _create_usd_dog_camera(camera_path: str) -> None:
    stage = omni.usd.get_context().get_stage()
    camera = UsdGeom.Camera.Define(stage, Sdf.Path(camera_path))
    camera.CreateFocalLengthAttr(DOG_CAMERA_FOCAL_LENGTH)
    camera.CreateFocusDistanceAttr(400.0)
    camera.CreateHorizontalApertureAttr(DOG_CAMERA_HORIZONTAL_APERTURE)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
    _set_xform(camera_path, DOG_CAMERA_POS, DOG_CAMERA_USD_ROT_WXYZ)


def _create_dog_camera_readers() -> dict[str, Camera]:
    camera_specs = {
        "agent_1": "/World/envs/env_0/Go2Dog1/base/front_camera",
        "agent_2": "/World/envs/env_0/Go2Dog2/base/front_camera",
    }
    cameras: dict[str, Camera] = {}
    for robot_id, camera_path in camera_specs.items():
        _create_usd_dog_camera(camera_path)
        cfg = CameraCfg(
            prim_path=camera_path,
            spawn=None,
            width=320,
            height=240,
            data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
            update_period=0.0,
        )
        cameras[robot_id] = Camera(cfg)
        print(f"[INFO] Rewrite USD dog camera reader: {robot_id} -> {camera_path}")
    return cameras


def _update_camera_readers(camera_readers: dict[str, Camera], dt: float) -> None:
    for camera in camera_readers.values():
        camera.update(dt)


def _update_imu_readers(imu_readers: dict[str, ImuSensor], dt: float) -> None:
    for imu in imu_readers.values():
        imu.update(dt)


def _update_lidar_readers(lidar_readers: dict[str, RayCaster], dt: float) -> None:
    for lidar in lidar_readers.values():
        lidar.update(dt)


def _reset_camera_readers(camera_readers: dict[str, Camera]) -> None:
    for camera in camera_readers.values():
        if hasattr(camera, "reset"):
            camera.reset()


def _scene_camera_prim_path(robot_id: str) -> str:
    if robot_id == "agent_1":
        return "/World/envs/env_0/Go2Dog1/base/front_camera"
    return "/World/envs/env_0/Go2Dog2/base/front_camera"


def _load_shared_slam_scene(scene_usd_path: Path) -> None:
    import omni.usd

    if not scene_usd_path.exists():
        raise FileNotFoundError(f"SLAM scene USDA file not found: {scene_usd_path}")

    prim_utils.create_prim("/World", "Xform")
    prim_utils.create_prim("/World/SlamScene", "Xform", usd_path=str(scene_usd_path))
    prim_utils.create_prim("/World/Actors", "Xform")

    stage = omni.usd.get_context().get_stage()
    referenced_physics_scene = stage.GetPrimAtPath("/World/SlamScene/PhysicsScene")
    if referenced_physics_scene.IsValid() and referenced_physics_scene.IsActive():
        # SimulationContext already owns the live physics scene for this stage.
        # Keep the referenced USDA physics scene inactive so PhysX does not see
        # multiple physics scenes stepping in parallel.
        referenced_physics_scene.SetActive(False)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.dt, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    center_x = MAP_BOUNDS["center_x"]
    center_y = MAP_BOUNDS["center_y"]
    span = max(MAP_BOUNDS["width"], MAP_BOUNDS["height"])
    sim.set_camera_view(
        eye=[center_x, center_y - span * 1.4, span * 0.95],
        target=[center_x, center_y, 0.0],
    )

    _load_shared_slam_scene(args_cli.scene_usd)
    print(f"[INFO] Loaded rewrite scene from {args_cli.scene_usd}")

    mesh_info = create_static_localization_mesh()
    print(
        f"[INFO] Rewrite localization mesh ready: {mesh_info['mesh_path']} "
        f"({mesh_info['num_vertices']} verts, {mesh_info['num_triangles']} tris)"
    )

    scene_cfg = SurveillanceSceneCfg()
    scene_cfg.dog1_lidar.debug_vis = args_cli.show_lidar
    scene_cfg.dog2_lidar.debug_vis = args_cli.show_lidar
    scene = InteractiveScene(scene_cfg)
    dog_cameras = _create_dog_camera_readers()

    sim.reset()
    if hasattr(scene, "reset"):
        scene.reset()

    dogs = {
        "agent_1": scene.articulations[ROBOT_NAME_MAP["agent_1"]],
        "agent_2": scene.articulations[ROBOT_NAME_MAP["agent_2"]],
    }
    intruder = scene.articulations[INTRUDER_SCENE_NAME]
    cctv_cameras = {camera_id: scene.sensors[camera_id] for camera_id in CCTV_NAMES}
    dog_lidars = {
        "agent_1": scene.sensors[ROBOT_LIDAR_MAP["agent_1"]],
        "agent_2": scene.sensors[ROBOT_LIDAR_MAP["agent_2"]],
    }
    dog_imus = {
        "agent_1": scene.sensors[ROBOT_IMU_MAP["agent_1"]],
        "agent_2": scene.sensors[ROBOT_IMU_MAP["agent_2"]],
    }
    recorder = None
    if args_cli.record_video:
        feed_order = [*CCTV_NAMES, "agent_1", "agent_2"]
        recorder = SimulationVideoRecorder(
            output_path=args_cli.video_output,
            fps=args_cli.video_fps,
            feed_order=feed_order,
        )
    _initialize_actor_baselines(dogs, intruder)
    for dog in dogs.values():
        dog.write_data_to_sim()
    intruder.write_data_to_sim()
    sim.step()
    for dog in dogs.values():
        dog.update(args_cli.dt)
    intruder.update(args_cli.dt)
    _reset_camera_readers(dog_cameras)
    _update_camera_readers(dog_cameras, args_cli.dt)
    _update_camera_readers(cctv_cameras, args_cli.dt)
    _update_lidar_readers(dog_lidars, args_cli.dt)
    _update_imu_readers(dog_imus, args_cli.dt)

    ros2_bridge = None if args_cli.disable_ros2 else IsaacSimRos2Bridge(
        topic_prefix=args_cli.topic_prefix,
        control_topic_prefix=args_cli.control_topic_prefix,
        robot_ids=["agent_1", "agent_2"],
        intruder_ids=[INTRUDER_TOPIC_NAME],
        max_command_age=args_cli.max_command_age,
    )

    if args_cli.view_camera != "world":
        _set_active_viewport_camera(_scene_camera_prim_path(args_cli.view_camera))

    omni.timeline.get_timeline_interface().play()
    print("[INFO] Rewritten environment scene ready.")

    wall_clock_start = time.monotonic()
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

        for robot_id, dog in dogs.items():
            if robot_id in low_level_applied:
                dog.write_data_to_sim()
                continue
            dog.set_joint_position_target(dog.data.default_joint_pos)
            dog.write_data_to_sim()
        _stabilize_suspect(intruder)
        intruder.set_joint_position_target(intruder.data.default_joint_pos)
        intruder.write_data_to_sim()
        sim.step()
        for dog in dogs.values():
            dog.update(args_cli.dt)
        intruder.update(args_cli.dt)
        _update_camera_readers(dog_cameras, args_cli.dt)
        _update_camera_readers(cctv_cameras, args_cli.dt)
        _update_lidar_readers(dog_lidars, args_cli.dt)
        _update_imu_readers(dog_imus, args_cli.dt)
        if recorder is not None:
            current_sim_time = (step_idx + 1) * args_cli.dt
            while current_sim_time + 1.0e-9 >= next_capture_sim_time:
                recorder.maybe_write(dog_cameras, cctv_cameras)
                next_capture_sim_time += capture_interval_sim

        if ros2_bridge is not None and step_idx % max(1, args_cli.publish_every) == 0:
            ros2_bridge.publish(dogs, intruder, dog_cameras, cctv_cameras, dog_lidars, dog_imus, step_idx)

        if step_idx in (0, effective_steps - 1):
            positions = {robot_id: dog.data.root_pos_w[0].detach().cpu().tolist() for robot_id, dog in dogs.items()}
            positions[INTRUDER_TOPIC_NAME] = intruder.data.root_pos_w[0].detach().cpu().tolist()
            print(f"[INFO] Step {step_idx + 1}/{effective_steps} actor positions: {positions}")
        if step_idx > 0 and step_idx % 200 == 0:
            sim_time = (step_idx + 1) * args_cli.dt
            wall_time = time.monotonic() - wall_clock_start
            rtf = sim_time / wall_time if wall_time > 1.0e-6 else 0.0
            print(f"[TIMING] step={step_idx} sim_time={sim_time:.3f}s wall_time={wall_time:.3f}s RTF={rtf:.4f}")

    if recorder is not None:
        recorder.close()

    if args_cli.keep_open:
        print("[INFO] Keeping Isaac Sim open. Close the window to stop.")
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
            for robot_id, dog in dogs.items():
                if robot_id in low_level_applied:
                    dog.write_data_to_sim()
                    continue
                dog.set_joint_position_target(dog.data.default_joint_pos)
                dog.write_data_to_sim()
            _stabilize_suspect(intruder)
            intruder.set_joint_position_target(intruder.data.default_joint_pos)
            intruder.write_data_to_sim()
            sim.step()
            for dog in dogs.values():
                dog.update(args_cli.dt)
            intruder.update(args_cli.dt)
            _update_camera_readers(dog_cameras, args_cli.dt)
            _update_camera_readers(cctv_cameras, args_cli.dt)
            _update_lidar_readers(dog_lidars, args_cli.dt)
            _update_imu_readers(dog_imus, args_cli.dt)
            if ros2_bridge is not None:
                ros2_bridge.publish(dogs, intruder, dog_cameras, cctv_cameras, dog_lidars, dog_imus, -1)

    if ros2_bridge is not None:
        ros2_bridge.shutdown()
    omni.timeline.get_timeline_interface().stop()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"[ERROR] Rewritten environment simulation failed: {exc}")
        raise
