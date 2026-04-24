#!/usr/bin/env python3
"""Validate the first Isaac Sim scene for the simulation module.

This script loads the imported SLAM scene, places two Unitree Go2 robots and one
humanoid intruder, then attaches one camera and one RTX LiDAR to each Go2.

It is intentionally standalone inside the Simulation module. It also publishes
the Simulation-Core ROS2 topic contract when launched through
scripts/launch_simulation.sh.
"""

from __future__ import annotations

import argparse
import json
import math
import time
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
parser.add_argument("--show-lidar", action="store_true", help="Enable RTX LiDAR point-cloud debug visualization.")
parser.add_argument("--disable-ros2", action="store_true", help="Do not publish Simulation topics to ROS2.")
parser.add_argument("--topic-prefix", default="/factory/simulation", help="ROS2 topic prefix used by Core.")
parser.add_argument("--control-topic-prefix", default="/factory/control", help="ROS2 control topic prefix used by Core.")
parser.add_argument("--publish-every", type=int, default=4, help="Publish ROS2 data every N simulation steps.")
parser.add_argument("--max-command-age", type=float, default=1.0, help="Ignore stale control commands after this many seconds.")
parser.add_argument("--command-scale", type=float, default=1.0, help="Scale incoming world-frame velocity commands.")
parser.add_argument(
    "--sensor-view",
    choices=["agent_1", "agent_2"],
    help="Shortcut for live inspection: switch to a dog camera and show its RTX LiDAR point cloud.",
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
import omni.kit.app
import omni.kit.commands
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
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab_assets.robots.unitree import H1_MINIMAL_CFG, UNITREE_GO2_CFG

try:
    from isaacsim.sensors.rtx import LidarRtx
except ImportError:
    LidarRtx = None

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Image, LaserScan, PointCloud2, PointField
    from std_msgs.msg import String
except ImportError:
    rclpy = None
    PoseStamped = None
    Image = None
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

INTRUDER = {
    "prim_path": "/World/Actors/intruder_1",
    "pos": (2.0, -0.5, 1.05),
    "yaw_deg": 180.0,
}

LIDAR_FLAT_SCAN_ANNOTATOR = "IsaacComputeRTXLidarFlatScan"
LIDAR_POINT_CLOUD_ANNOTATOR = "IsaacExtractRTXSensorPointCloudNoAccumulator"


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
        lidar_readers: dict[str, LidarRtx],
        step_idx: int,
    ) -> None:
        stamp = self.node.get_clock().now().to_msg()
        robot_states = {}
        for robot_id in self.robot_ids:
            pos = _root_position(dogs[robot_id])
            robot_states[robot_id] = pos
            self.robot_pose_pubs[robot_id].publish(self._make_pose(robot_id, pos, stamp, _root_quat(dogs[robot_id])))
            self.locomotion_observation_pubs[robot_id].publish(
                String(data=json.dumps(_make_locomotion_observation(robot_id, dogs[robot_id], step_idx)))
            )
            camera_msg = self._make_camera_image(robot_id, stamp, camera_readers.get(robot_id))
            if camera_msg is not None:
                self.camera_pubs[robot_id].publish(camera_msg)
            depth_msg = self._make_depth_image(robot_id, stamp, camera_readers.get(robot_id))
            if depth_msg is not None:
                self.depth_pubs[robot_id].publish(depth_msg)
            lidar = lidar_readers.get(robot_id)
            scan_msg = self._make_scan(robot_id, stamp, lidar)
            if scan_msg is not None:
                self.lidar_pubs[robot_id].publish(scan_msg)
            point_cloud_msg = self._make_point_cloud(robot_id, stamp, lidar)
            if point_cloud_msg is not None:
                self.lidar_point_cloud_pubs[robot_id].publish(point_cloud_msg)

        intruder_pos = _root_position(intruder)
        intruder_states = {"intruder_1": intruder_pos}
        self.intruder_pose_pubs["intruder_1"].publish(
            self._make_pose("intruder_1", intruder_pos, stamp, _root_quat(intruder))
        )
        self.state_pub.publish(String(data=json.dumps(self._make_state(robot_states, intruder_states, step_idx))))

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
        msg.header.frame_id = f"{robot_id}/front_camera"
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
    def _make_scan(robot_id: str, stamp, lidar: LidarRtx | None) -> LaserScan | None:
        if lidar is None:
            return None
        frame = lidar.get_current_frame()
        flat_scan = frame.get(LIDAR_FLAT_SCAN_ANNOTATOR, {})
        ranges = _extract_lidar_ranges(flat_scan)
        if ranges is None:
            return None

        angle_min, angle_max = _extract_lidar_angle_range(flat_scan, len(ranges))
        range_min, range_max = _extract_lidar_depth_range(flat_scan)
        intensities = _extract_lidar_intensities(flat_scan, len(ranges))

        msg = LaserScan()
        msg.header.stamp = stamp
        msg.header.frame_id = f"{robot_id}/front_lidar"
        msg.angle_min = angle_min
        msg.angle_max = angle_max
        msg.angle_increment = (angle_max - angle_min) / max(1, len(ranges) - 1)
        msg.time_increment = 0.0
        msg.scan_time = 0.1
        msg.range_min = range_min
        msg.range_max = range_max
        msg.ranges = ranges.astype(float).tolist()
        msg.intensities = intensities.astype(float).tolist()
        return msg

    @staticmethod
    def _make_point_cloud(robot_id: str, stamp, lidar: LidarRtx | None) -> PointCloud2 | None:
        if lidar is None:
            return None
        frame = lidar.get_current_frame()
        point_cloud = frame.get(LIDAR_POINT_CLOUD_ANNOTATOR, {})
        points = _extract_lidar_point_cloud(point_cloud)
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
            "robots": {
                robot_id: {"position": list(pos)}
                for robot_id, pos in robot_states.items()
            },
            "intruders": {
                intruder_id: {"position": list(pos)}
                for intruder_id, pos in intruder_states.items()
            },
        }


def _enable_extension(ext_name: str) -> None:
    """Enable an Isaac Sim extension if it is available."""
    manager = omni.kit.app.get_app().get_extension_manager()
    if manager.get_enabled_extension_id(ext_name):
        return
    manager.set_extension_enabled_immediate(ext_name, True)


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


def _set_xform(
    prim_path: str,
    translation: tuple[float, float, float],
    yaw_deg: float = 0.0,
    quat_wxyz: tuple[float, float, float, float] | None = None,
) -> None:
    """Set translate + orient xform ops on an existing prim."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Cannot set transform. Prim does not exist: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translation))
    xformable.AddOrientOp().Set(_make_quat_xyzw(quat_wxyz) if quat_wxyz else _make_quat_z(yaw_deg))
    xformable.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))


def _create_usd_camera(camera_path: str) -> None:
    """Create a USD pinhole camera under a robot base link."""
    stage = omni.usd.get_context().get_stage()
    camera = UsdGeom.Camera.Define(stage, Sdf.Path(camera_path))
    camera.CreateFocalLengthAttr(24.0)
    camera.CreateFocusDistanceAttr(400.0)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
    _set_xform(camera_path, (0.34, 0.0, 0.12), quat_wxyz=(0.5, 0.5, -0.5, -0.5))


def _create_rtx_lidar(lidar_path: str, parent_path: str) -> str:
    """Create an RTX LiDAR under a robot base link."""
    _enable_extension("omni.isaac.sensor")
    created = omni.kit.commands.execute(
        "IsaacSensorCreateRtxLidar",
        path=lidar_path,
        parent=None,
        config="RPLIDAR_S2E",
        translation=Gf.Vec3d(0.26, 0.0, 0.18),
        orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
        visibility=True,
    )
    prim = created[1] if isinstance(created, tuple) else created
    if prim is None or not prim.IsValid():
        raise RuntimeError(f"Failed to create RTX LiDAR under {parent_path}")
    omni_lidar_path = _find_omni_lidar_path(prim.GetPath().pathString)
    if omni_lidar_path is None:
        raise RuntimeError(
            f"RTX LiDAR command did not create an OmniLidar at or under {prim.GetPath().pathString}"
        )
    return omni_lidar_path


def _find_omni_lidar_path(root_path: str) -> str | None:
    """Return the first OmniLidar prim at or below root_path."""
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return None
    if root_prim.GetTypeName() == "OmniLidar":
        return root_prim.GetPath().pathString

    prims_to_visit = list(root_prim.GetChildren())
    while prims_to_visit:
        prim = prims_to_visit.pop(0)
        if prim.GetTypeName() == "OmniLidar":
            return prim.GetPath().pathString
        prims_to_visit.extend(list(prim.GetChildren()))
    return None


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


def _create_lidar_readers(lidar_paths: dict[str, str], show_visualization: bool = False) -> dict[str, LidarRtx]:
    """Create RTX LiDAR readers and attach real scan / point-cloud annotators."""
    if LidarRtx is None:
        print("[WARNING] LidarRtx API is unavailable. LiDAR ROS2 topics will not be published.")
        return {}

    lidar_readers: dict[str, LidarRtx] = {}
    for dog_id, lidar_path in lidar_paths.items():
        lidar = LidarRtx(prim_path=lidar_path, name=f"{dog_id}_front_lidar")
        lidar.initialize()
        lidar.attach_annotator(LIDAR_FLAT_SCAN_ANNOTATOR)
        lidar.attach_annotator(LIDAR_POINT_CLOUD_ANNOTATOR)
        if show_visualization:
            lidar.enable_visualization()
        lidar_readers[dog_id] = lidar
        print(f"[INFO] {dog_id} RTX LiDAR reader: {lidar_path}")
    return lidar_readers


def _print_lidar_status(lidar_readers: dict[str, LidarRtx], step_idx: int) -> None:
    """Print a small heartbeat so we know LiDAR frames are being updated."""
    for robot_id, lidar in lidar_readers.items():
        frame = lidar.get_current_frame()
        point_cloud = _extract_lidar_point_cloud(frame.get(LIDAR_POINT_CLOUD_ANNOTATOR, {}))
        ranges = _extract_lidar_ranges(frame.get(LIDAR_FLAT_SCAN_ANNOTATOR, {}))
        point_count = 0 if point_cloud is None else point_cloud.shape[0]
        range_count = 0 if ranges is None else ranges.size
        print(f"[INFO] Step {step_idx}: {robot_id} lidar ranges={range_count} points={point_count}")


def _extract_lidar_ranges(flat_scan: dict) -> np.ndarray | None:
    raw_depth = flat_scan.get("linearDepthData")
    if raw_depth is None:
        return None

    depths = np.asarray(raw_depth, dtype=np.float32).reshape(-1)
    if depths.size == 0:
        return None

    num_cols = int(flat_scan.get("numCols") or depths.size)
    num_rows = int(flat_scan.get("numRows") or max(1, depths.size // max(1, num_cols)))
    if num_cols > 0 and num_rows > 0 and depths.size == num_rows * num_cols:
        depths = depths.reshape(num_rows, num_cols)
        valid = np.isfinite(depths) & (depths > 0.0)
        ranges = np.full((num_cols,), np.inf, dtype=np.float32)
        if valid.any():
            ranges = np.where(valid, depths, np.inf).min(axis=0).astype(np.float32)
    else:
        ranges = depths.astype(np.float32)

    range_min, range_max = _extract_lidar_depth_range(flat_scan)
    valid_ranges = np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max)
    return np.where(valid_ranges, ranges, np.inf).astype(np.float32)


def _extract_lidar_intensities(flat_scan: dict, count: int) -> np.ndarray:
    raw_intensities = flat_scan.get("intensitiesData")
    if raw_intensities is None:
        return np.zeros((count,), dtype=np.float32)

    intensities = np.asarray(raw_intensities, dtype=np.float32).reshape(-1)
    num_cols = int(flat_scan.get("numCols") or count)
    num_rows = int(flat_scan.get("numRows") or max(1, intensities.size // max(1, num_cols)))
    if num_cols > 0 and num_rows > 0 and intensities.size == num_rows * num_cols:
        intensities = intensities.reshape(num_rows, num_cols).max(axis=0)

    if intensities.size < count:
        padded = np.zeros((count,), dtype=np.float32)
        padded[: intensities.size] = intensities
        return padded
    return intensities[:count].astype(np.float32)


def _extract_lidar_angle_range(flat_scan: dict, count: int) -> tuple[float, float]:
    azimuth_range = flat_scan.get("azimuthRange")
    if azimuth_range is None or len(azimuth_range) < 2:
        return (-math.pi, math.pi)

    angle_min = float(azimuth_range[0])
    angle_max = float(azimuth_range[1])
    if abs(angle_min) > 2.0 * math.pi or abs(angle_max) > 2.0 * math.pi:
        angle_min = math.radians(angle_min)
        angle_max = math.radians(angle_max)
    if count > 1 and angle_max <= angle_min:
        angle_max = angle_min + 2.0 * math.pi
    return angle_min, angle_max


def _extract_lidar_depth_range(flat_scan: dict) -> tuple[float, float]:
    depth_range = flat_scan.get("depthRange")
    if depth_range is None or len(depth_range) < 2:
        return (0.05, 30.0)
    return float(depth_range[0]), float(depth_range[1])


def _extract_lidar_point_cloud(point_cloud: dict) -> np.ndarray | None:
    raw_points = point_cloud.get("data") if isinstance(point_cloud, dict) else None
    if raw_points is None:
        return None

    points = np.asarray(raw_points, dtype=np.float32)
    if points.size == 0:
        return None
    if points.ndim == 1:
        if points.size % 3 != 0:
            return None
        points = points.reshape(-1, 3)
    else:
        points = points.reshape(-1, points.shape[-1])
    if points.shape[1] < 3:
        return None
    points = points[:, :3]
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    return points.astype(np.float32) if points.size else None


def _load_environment(scene_usd: Path) -> None:
    """Reference the pure scene under /World/SlamScene."""
    if not scene_usd.exists():
        raise FileNotFoundError(f"Scene file does not exist: {scene_usd}")

    prim_utils.create_prim("/World", "Xform")
    prim_utils.create_prim("/World/SlamScene", "Xform", usd_path=str(scene_usd))
    prim_utils.create_prim("/World/Actors", "Xform")

    # The imported pure scene already contains a PhysicsScene. SimulationContext
    # owns the active physics scene for this validation script, so we deactivate
    # the referenced one to avoid per-scene stepping warnings.
    stage = omni.usd.get_context().get_stage()
    referenced_physics_scene = stage.GetPrimAtPath("/World/SlamScene/PhysicsScene")
    if referenced_physics_scene.IsValid():
        referenced_physics_scene.SetActive(False)


def _spawn_actors() -> tuple[dict[str, Articulation], Articulation]:
    """Spawn two Go2 robots and one humanoid intruder."""
    dogs: dict[str, Articulation] = {}
    for dog_id, spec in DOGS.items():
        dogs[dog_id] = Articulation(UNITREE_GO2_CFG.replace(prim_path=spec["prim_path"]))

    intruder = Articulation(H1_MINIMAL_CFG.replace(prim_path=INTRUDER["prim_path"]))
    return dogs, intruder


def _attach_sensors(dog_specs: dict[str, dict]) -> tuple[list[str], dict[str, str]]:
    """Attach camera and LiDAR prims to each dog."""
    sensor_paths: list[str] = []
    lidar_paths: dict[str, str] = {}
    for dog_id, spec in dog_specs.items():
        base_path = f"{spec['prim_path']}/base"
        camera_path = f"{base_path}/front_camera"
        lidar_path = f"{base_path}/front_lidar"

        _create_usd_camera(camera_path)
        actual_lidar_path = _create_rtx_lidar(lidar_path, base_path)
        lidar_paths[dog_id] = actual_lidar_path

        sensor_paths.extend([camera_path, actual_lidar_path])
        print(f"[INFO] {dog_id} camera: {camera_path}")
        print(f"[INFO] {dog_id} RTX LiDAR: {actual_lidar_path}")
    return sensor_paths, lidar_paths


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
            data_types=["rgb", "distance_to_image_plane"],
            update_period=0.0,
        )
        cameras[dog_id] = Camera(cfg)
        print(f"[INFO] {dog_id} RGB camera reader: {camera_path}")
    return cameras


def _update_camera_readers(camera_readers: dict[str, Camera], dt: float) -> None:
    for camera in camera_readers.values():
        camera.update(dt)


def _write_default_state(entity: Articulation, pos: tuple[float, float, float]) -> None:
    """Place an articulation root at a world position and write default joints."""
    root_state = entity.data.default_root_state.clone()
    root_state[:, :3] = torch.tensor(pos, device=root_state.device, dtype=root_state.dtype)
    entity.write_root_state_to_sim(root_state)
    entity.write_joint_state_to_sim(entity.data.default_joint_pos.clone(), entity.data.default_joint_vel.clone())
    entity.reset()


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
    dogs, intruder = _spawn_actors()
    sensor_paths, lidar_paths = _attach_sensors(DOGS)
    camera_readers = _create_camera_readers(DOGS)

    expected_prims = [
        "/World/SlamScene",
        "/World/Actors",
        INTRUDER["prim_path"],
        *[spec["prim_path"] for spec in DOGS.values()],
        *sensor_paths,
    ]
    _validate_prims(expected_prims)

    sim.reset()
    _initialize_actor_states(dogs, intruder)
    lidar_readers = _create_lidar_readers(lidar_paths, show_visualization=args_cli.show_lidar)
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

    for step_idx in range(args_cli.steps):
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
        intruder.set_joint_position_target(intruder.data.default_joint_pos)
        intruder.write_data_to_sim()

        sim.step()

        for dog in dogs.values():
            dog.update(args_cli.dt)
        intruder.update(args_cli.dt)
        _update_camera_readers(camera_readers, args_cli.dt)
        if ros2_bridge is not None and step_idx % max(1, args_cli.publish_every) == 0:
            ros2_bridge.publish(dogs, intruder, camera_readers, lidar_readers, step_idx)

        if step_idx in (0, args_cli.steps - 1):
            positions = {
                dog_id: dog.data.root_pos_w[0].detach().cpu().tolist()
                for dog_id, dog in dogs.items()
            }
            positions["intruder_1"] = intruder.data.root_pos_w[0].detach().cpu().tolist()
            print(f"[INFO] Step {step_idx + 1}/{args_cli.steps} actor positions: {positions}")
        if lidar_readers and step_idx in (0, args_cli.steps - 1):
            _print_lidar_status(lidar_readers, step_idx + 1)

    if not args_cli.no_export:
        _export_stage(args_cli.export_usd)

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
            intruder.set_joint_position_target(intruder.data.default_joint_pos)
            intruder.write_data_to_sim()
            sim.step()
            for dog in dogs.values():
                dog.update(args_cli.dt)
            intruder.update(args_cli.dt)
            _update_camera_readers(camera_readers, args_cli.dt)
            if ros2_bridge is not None and keep_open_step % max(1, args_cli.publish_every) == 0:
                ros2_bridge.publish(dogs, intruder, camera_readers, lidar_readers, keep_open_step)
            keep_open_step += 1
            for lidar in lidar_readers.values():
                lidar.get_current_frame()

    if ros2_bridge is not None:
        ros2_bridge.shutdown()

    print("[INFO] Validation complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        carb.log_error(f"Simulation validation failed: {exc}")
        raise
    finally:
        simulation_app.close()
