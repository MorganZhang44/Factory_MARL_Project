from __future__ import annotations

import asyncio
import base64
import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

import rclpy
import uvicorn
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from factory_core.state_mirror import StateMirror
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, LaserScan, PointCloud2
from std_msgs.msg import String


DEFAULT_ROBOTS = ["agent_1", "agent_2"]
DEFAULT_INTRUDERS = ["intruder_1"]
DEFAULT_CCTV_IDS = ["cam_nw", "cam_ne", "cam_e_upper", "cam_e_lower", "cam_se", "cam_sw"]


@dataclass
class EntityState:
    pose: PoseStamped | None = None


@dataclass
class SensorState:
    camera_frame_id: str | None = None
    camera_stamp_sec: int | None = None
    camera_msg: Image | None = None
    camera_payload: dict[str, Any] | None = None
    depth_frame_id: str | None = None
    depth_stamp_sec: int | None = None
    depth_msg: Image | None = None
    depth_payload: dict[str, Any] | None = None
    semantic_frame_id: str | None = None
    semantic_stamp_sec: int | None = None
    semantic_msg: Image | None = None
    semantic_payload: dict[str, Any] | None = None
    imu_frame_id: str | None = None
    imu_stamp_sec: int | None = None
    imu_msg: Imu | None = None
    imu_payload: dict[str, Any] | None = None
    locomotion_observation: dict[str, Any] | None = None
    lidar_frame_id: str | None = None
    lidar_stamp_sec: int | None = None
    lidar_sample_count: int = 0
    lidar_msg: LaserScan | None = None
    lidar_payload: dict[str, Any] | None = None
    lidar_point_cloud_frame_id: str | None = None
    lidar_point_cloud_stamp_sec: int | None = None
    lidar_point_cloud_count: int = 0
    lidar_point_cloud_msg: PointCloud2 | None = None
    lidar_point_cloud_payload: dict[str, Any] | None = None


@dataclass
class CctvState:
    camera_msg: Image | None = None
    camera_payload: dict[str, Any] | None = None
    semantic_msg: Image | None = None
    semantic_payload: dict[str, Any] | None = None


@dataclass
class CoreState:
    robots: dict[str, EntityState] = field(default_factory=dict)
    intruders: dict[str, EntityState] = field(default_factory=dict)
    sensors: dict[str, SensorState] = field(default_factory=dict)
    cctv: dict[str, CctvState] = field(default_factory=dict)
    latest_simulation_state: dict[str, Any] | None = None


class CoreControlNode(Node):
    """ROS2-facing core control layer.

    Version 1 only receives and normalizes simulation data. Later this node can
    route state to perception, decision, planning, and locomotion modules.
    """

    def __init__(self) -> None:
        super().__init__("factory_core_control")
        self._state_group = MutuallyExclusiveCallbackGroup()
        self._control_group = MutuallyExclusiveCallbackGroup()

        self.declare_parameter("robot_ids", DEFAULT_ROBOTS)
        self.declare_parameter("intruder_ids", DEFAULT_INTRUDERS)
        self.declare_parameter("cctv_ids", DEFAULT_CCTV_IDS)
        self.declare_parameter("topic_prefix", "/factory/simulation")
        self.declare_parameter("heartbeat_period", 1.0)
        self.declare_parameter("stale_after", 1.0)
        self.declare_parameter("state_host", "0.0.0.0")
        self.declare_parameter("state_port", 8765)
        self.declare_parameter("state_websocket_period", 0.1)
        self.declare_parameter("enable_control_loop", True)
        self.declare_parameter("control_topic_prefix", "/factory/control")
        self.declare_parameter("control_period", 0.02)
        self.declare_parameter("planning_period", 0.5)
        self.declare_parameter("path_stale_after", 2.0)
        self.declare_parameter("navdp_timeout", 10.0)
        self.declare_parameter("locomotion_timeout", 0.08)
        self.declare_parameter("perception_timeout", 0.4)
        self.declare_parameter("marl_timeout", 0.15)
        self.declare_parameter("navdp_url", "http://127.0.0.1:8889")
        self.declare_parameter("locomotion_url", "http://127.0.0.1:8890")
        self.declare_parameter("perception_url", "http://127.0.0.1:8891")
        self.declare_parameter("marl_url", "http://127.0.0.1:8892")
        self.declare_parameter("perception_period", 0.04)
        self.declare_parameter("perception_record_dir", "")
        self.declare_parameter("marl_period", 0.1)
        self.declare_parameter("use_perception_output", True)
        self.declare_parameter("use_marl_output", True)
        self.declare_parameter("simulation_dt", 0.005)
        self.robot_ids = list(self.get_parameter("robot_ids").value)
        self.intruder_ids = list(self.get_parameter("intruder_ids").value)
        self.cctv_ids = list(self.get_parameter("cctv_ids").value)
        self.topic_prefix = str(self.get_parameter("topic_prefix").value).rstrip("/")
        heartbeat_period = float(self.get_parameter("heartbeat_period").value)
        stale_after = float(self.get_parameter("stale_after").value)
        self.state_host = str(self.get_parameter("state_host").value)
        self.state_port = int(self.get_parameter("state_port").value)
        self.state_websocket_period = float(self.get_parameter("state_websocket_period").value)
        self.enable_control_loop = bool(self.get_parameter("enable_control_loop").value)
        self.control_topic_prefix = str(self.get_parameter("control_topic_prefix").value).rstrip("/")
        self.control_period = float(self.get_parameter("control_period").value)
        self.planning_period = float(self.get_parameter("planning_period").value)
        self.path_stale_after = float(self.get_parameter("path_stale_after").value)
        self.navdp_timeout = float(self.get_parameter("navdp_timeout").value)
        self.locomotion_timeout = float(self.get_parameter("locomotion_timeout").value)
        self.perception_timeout = float(self.get_parameter("perception_timeout").value)
        self.marl_timeout = float(self.get_parameter("marl_timeout").value)
        self.navdp_url = str(self.get_parameter("navdp_url").value).rstrip("/")
        self.locomotion_url = str(self.get_parameter("locomotion_url").value).rstrip("/")
        self.perception_url = str(self.get_parameter("perception_url").value).rstrip("/")
        self.marl_url = str(self.get_parameter("marl_url").value).rstrip("/")
        self.perception_period = float(self.get_parameter("perception_period").value)
        record_dir_value = str(self.get_parameter("perception_record_dir").value).strip()
        self.marl_period = float(self.get_parameter("marl_period").value)
        self.use_perception_output = bool(self.get_parameter("use_perception_output").value)
        self.use_marl_output = bool(self.get_parameter("use_marl_output").value)
        self.simulation_dt = float(self.get_parameter("simulation_dt").value)
        self.perception_record_dir = Path(record_dir_value).expanduser() if record_dir_value else None

        self.state = CoreState(
            robots={robot_id: EntityState() for robot_id in self.robot_ids},
            intruders={intruder_id: EntityState() for intruder_id in self.intruder_ids},
            sensors={robot_id: SensorState() for robot_id in self.robot_ids},
            cctv={camera_id: CctvState() for camera_id in self.cctv_ids},
        )
        self.state_mirror = StateMirror(
            self.robot_ids,
            self.intruder_ids,
            stale_after,
            self.topic_prefix,
            cctv_ids=self.cctv_ids,
        )
        self.motion_command_pub = self.create_publisher(
            String,
            f"{self.control_topic_prefix}/locomotion/motion_command",
            10,
        )
        self._last_control_warning = 0.0
        self._plan_cache: dict[str, dict[str, Any]] = {}
        self._planning_inflight: set[str] = set()
        self._plan_lock = threading.Lock()
        self._last_control_step = -1
        self._last_locomotion_sim_time: dict[str, float] = {robot_id: float("-inf") for robot_id in self.robot_ids}
        self._perception_cache: dict[str, Any] | None = None
        self._perception_inflight = False
        self._perception_lock = threading.Lock()
        self._perception_record_lock = threading.Lock()
        self._marl_cache: dict[str, Any] | None = None
        self._marl_inflight = False
        self._marl_lock = threading.Lock()
        self._last_marl_debug_log = 0.0

        if self.perception_record_dir is not None:
            self.perception_record_dir.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f"Perception recording enabled: {self.perception_record_dir}")

        self.create_subscription(
            String,
            f"{self.topic_prefix}/state",
            self._on_simulation_state,
            10,
            callback_group=self._state_group,
        )

        for robot_id in self.robot_ids:
            self.create_subscription(
                PoseStamped,
                f"{self.topic_prefix}/{robot_id}/pose",
                lambda msg, rid=robot_id: self._on_robot_pose(rid, msg),
                20,
                callback_group=self._state_group,
            )
            self.create_subscription(
                Image,
                f"{self.topic_prefix}/{robot_id}/camera/image_raw",
                lambda msg, rid=robot_id: self._on_camera(rid, msg),
                5,
                callback_group=self._state_group,
            )
            self.create_subscription(
                Image,
                f"{self.topic_prefix}/{robot_id}/camera/depth",
                lambda msg, rid=robot_id: self._on_depth(rid, msg),
                5,
                callback_group=self._state_group,
            )
            self.create_subscription(
                Image,
                f"{self.topic_prefix}/{robot_id}/camera/semantic_segmentation",
                lambda msg, rid=robot_id: self._on_semantic(rid, msg),
                5,
                callback_group=self._state_group,
            )
            self.create_subscription(
                Imu,
                f"{self.topic_prefix}/{robot_id}/imu",
                lambda msg, rid=robot_id: self._on_imu(rid, msg),
                20,
                callback_group=self._state_group,
            )
            self.create_subscription(
                LaserScan,
                f"{self.topic_prefix}/{robot_id}/lidar/scan",
                lambda msg, rid=robot_id: self._on_lidar(rid, msg),
                10,
                callback_group=self._state_group,
            )
            self.create_subscription(
                PointCloud2,
                f"{self.topic_prefix}/{robot_id}/lidar/points",
                lambda msg, rid=robot_id: self._on_lidar_points(rid, msg),
                5,
                callback_group=self._state_group,
            )
            self.create_subscription(
                String,
                f"{self.topic_prefix}/{robot_id}/locomotion/observation",
                lambda msg, rid=robot_id: self._on_locomotion_observation(rid, msg),
                20,
                callback_group=self._state_group,
            )

        for camera_id in self.cctv_ids:
            self.create_subscription(
                Image,
                f"{self.topic_prefix}/cctv/{camera_id}/image_raw",
                lambda msg, cid=camera_id: self._on_cctv_camera(cid, msg),
                5,
                callback_group=self._state_group,
            )
            self.create_subscription(
                Image,
                f"{self.topic_prefix}/cctv/{camera_id}/semantic_segmentation",
                lambda msg, cid=camera_id: self._on_cctv_semantic(cid, msg),
                5,
                callback_group=self._state_group,
            )

        for intruder_id in self.intruder_ids:
            self.create_subscription(
                PoseStamped,
                f"{self.topic_prefix}/{intruder_id}/pose",
                lambda msg, iid=intruder_id: self._on_intruder_pose(iid, msg),
                20,
                callback_group=self._state_group,
            )

        self.create_timer(heartbeat_period, self._heartbeat, callback_group=self._state_group)
        if self.enable_control_loop:
            self.create_timer(self.control_period, self._control_tick, callback_group=self._control_group)
        self._server = None
        self._server_thread = threading.Thread(target=self._run_state_server, daemon=True)
        self._server_thread.start()
        self.get_logger().info(
            "Core control node listening under "
            f"{self.topic_prefix} for robots={self.robot_ids}, intruders={self.intruder_ids}"
        )
        self.get_logger().info(f"Core state API listening on http://{self.state_host}:{self.state_port}")
        if self.enable_control_loop:
            self.get_logger().info(
                "Core control loop enabled: "
                f"navdp={self.navdp_url}, locomotion={self.locomotion_url}, "
                f"control_period={self.control_period}s, planning_period={self.planning_period}s, "
                f"use_perception_output={self.use_perception_output}, "
                f"use_marl_output={self.use_marl_output}, "
                f"command_topic={self.control_topic_prefix}/locomotion/motion_command"
            )

    def _on_simulation_state(self, msg: String) -> None:
        try:
            self.state.latest_simulation_state = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Invalid simulation state JSON: {exc}")
        self.state_mirror.update_aggregate(msg.data, time.monotonic())

    def _on_robot_pose(self, robot_id: str, msg: PoseStamped) -> None:
        self.state.robots[robot_id].pose = msg
        self.state_mirror.update_robot_pose(robot_id, msg, time.monotonic())

    def _on_intruder_pose(self, intruder_id: str, msg: PoseStamped) -> None:
        self.state.intruders[intruder_id].pose = msg
        self.state_mirror.update_intruder_pose(intruder_id, msg, time.monotonic())

    def _on_camera(self, robot_id: str, msg: Image) -> None:
        sensor = self.state.sensors[robot_id]
        sensor.camera_frame_id = msg.header.frame_id
        sensor.camera_stamp_sec = int(msg.header.stamp.sec)
        sensor.camera_msg = msg
        sensor.camera_payload = self._image_payload(msg)
        self.state_mirror.update_camera(robot_id, msg, time.monotonic())

    def _on_depth(self, robot_id: str, msg: Image) -> None:
        sensor = self.state.sensors[robot_id]
        sensor.depth_frame_id = msg.header.frame_id
        sensor.depth_stamp_sec = int(msg.header.stamp.sec)
        sensor.depth_msg = msg
        sensor.depth_payload = self._image_payload(msg)
        self.state_mirror.update_depth(robot_id, msg, time.monotonic())

    def _on_semantic(self, robot_id: str, msg: Image) -> None:
        sensor = self.state.sensors[robot_id]
        sensor.semantic_frame_id = msg.header.frame_id
        sensor.semantic_stamp_sec = int(msg.header.stamp.sec)
        sensor.semantic_msg = msg
        sensor.semantic_payload = self._image_payload(msg)
        self.state_mirror.update_semantic(robot_id, msg, time.monotonic())

    def _on_cctv_camera(self, camera_id: str, msg: Image) -> None:
        self.state.cctv[camera_id].camera_msg = msg
        self.state.cctv[camera_id].camera_payload = self._image_payload(msg)
        self.state_mirror.update_cctv_camera(camera_id, msg, time.monotonic())

    def _on_cctv_semantic(self, camera_id: str, msg: Image) -> None:
        self.state.cctv[camera_id].semantic_msg = msg
        self.state.cctv[camera_id].semantic_payload = self._image_payload(msg)
        self.state_mirror.update_cctv_semantic(camera_id, msg, time.monotonic())

    def _on_imu(self, robot_id: str, msg: Imu) -> None:
        sensor = self.state.sensors[robot_id]
        sensor.imu_frame_id = msg.header.frame_id
        sensor.imu_stamp_sec = int(msg.header.stamp.sec)
        sensor.imu_msg = msg
        sensor.imu_payload = self._imu_payload(msg)
        self.state_mirror.update_imu(robot_id, msg, time.monotonic())

    def _on_locomotion_observation(self, robot_id: str, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            self.state.sensors[robot_id].locomotion_observation = payload
            self.state_mirror.update_locomotion_observation(robot_id, payload, time.monotonic())
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Invalid locomotion observation JSON for {robot_id}: {exc}")

    def _on_lidar(self, robot_id: str, msg: LaserScan) -> None:
        sensor = self.state.sensors[robot_id]
        sensor.lidar_frame_id = msg.header.frame_id
        sensor.lidar_stamp_sec = int(msg.header.stamp.sec)
        sensor.lidar_sample_count = len(msg.ranges)
        sensor.lidar_msg = msg
        sensor.lidar_payload = self._laser_scan_payload(msg)
        self.state_mirror.update_lidar(robot_id, msg, time.monotonic())

    def _on_lidar_points(self, robot_id: str, msg: PointCloud2) -> None:
        sensor = self.state.sensors[robot_id]
        sensor.lidar_point_cloud_frame_id = msg.header.frame_id
        sensor.lidar_point_cloud_stamp_sec = int(msg.header.stamp.sec)
        sensor.lidar_point_cloud_count = int(msg.width) * int(msg.height)
        sensor.lidar_point_cloud_msg = msg
        sensor.lidar_point_cloud_payload = self._point_cloud_payload(msg)
        self.state_mirror.update_lidar_points(robot_id, msg, time.monotonic())

    def _run_state_server(self) -> None:
        app = create_state_app(self.state_mirror, self.state_websocket_period)
        config = uvicorn.Config(app, host=self.state_host, port=self.state_port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._server.run()

    def _control_tick(self) -> None:
        sim_clock = self._current_sim_clock()
        if sim_clock is None:
            return
        current_step, current_sim_time = sim_clock
        if current_step <= self._last_control_step:
            return
        self._last_control_step = current_step

        self._maybe_start_perception()
        self._maybe_start_marl()
        perception = self._latest_perception()
        marl_result = self._latest_marl()
        intruder_id = self.intruder_ids[0] if self.intruder_ids else None
        intruder_pose = self.state.intruders.get(intruder_id).pose if intruder_id else None
        intruder_xy = self._perception_intruder_xy(perception) if self.use_perception_output else None
        if intruder_xy is None:
            intruder_xy = self._pose_to_xy(intruder_pose)
        if intruder_xy is None:
            return

        commands: dict[str, dict[str, Any]] = {}
        for robot_id, entity in self.state.robots.items():
            actual_robot_xy = self._pose_to_xy(entity.pose)
            if actual_robot_xy is None:
                continue

            robot_state = self._perception_robot_state(robot_id, perception) if self.use_perception_output else None
            if robot_state is None:
                robot_state = {"position": actual_robot_xy, "velocity": [0.0, 0.0]}
            sensor_payload = self._sensor_payload(robot_id)
            marl_subgoal = self._marl_subgoal_xy(marl_result, robot_id) if self.use_marl_output else None
            subgoal = {
                "robot_id": robot_id,
                "subgoal": marl_subgoal if marl_subgoal is not None else intruder_xy,
                "mode": "marl" if marl_subgoal is not None else "chase",
                "priority": 1,
            }
            robot_yaw = self._pose_yaw(entity.pose)
            local_goal = self._world_point_to_body_xy(entity.pose, subgoal["subgoal"], actual_robot_xy)
            self._maybe_start_planning(
                robot_id,
                robot_state,
                subgoal,
                sensor_payload,
                local_goal,
                robot_yaw,
                current_sim_time,
            )
            path = self._latest_path(robot_id, current_sim_time)
            if path is None:
                continue
            if current_sim_time - self._last_locomotion_sim_time.get(robot_id, float("-inf")) < self.control_period:
                continue
            try:
                body_command = self._body_command_from_navdp_path(entity.pose, actual_robot_xy, path)
                motion = self._post_json(
                    f"{self.locomotion_url}/command",
                    {
                        "robot_id": robot_id,
                        "robot_state": robot_state,
                        "path": path,
                        "body_velocity_command": body_command,
                        "locomotion_observation": self.state.sensors[robot_id].locomotion_observation,
                        "simulation_state": self.state.latest_simulation_state or {},
                    },
                    timeout=self.locomotion_timeout,
                )
            except (TimeoutError, HTTPError, URLError, json.JSONDecodeError, OSError, ValueError) as exc:
                self._warn_control_once(f"Control module call failed: {exc}")
                continue

            velocity = motion.get("velocity")
            if not self._valid_velocity(velocity):
                self._warn_control_once(f"Invalid locomotion velocity for {robot_id}: {velocity}")
                continue
            commands[robot_id] = {
                "velocity": [float(velocity[0]), float(velocity[1])],
                "action": motion.get("action"),
                "action_scale": motion.get("action_scale"),
                "subgoal": subgoal["subgoal"],
                "body_velocity_command": body_command,
                "path": path.get("waypoints", []),
            }
            self.state_mirror.update_locomotion_output(
                robot_id,
                locomotion={
                    "velocity": [float(velocity[0]), float(velocity[1])],
                    "body_velocity_command": body_command,
                    "action": motion.get("action"),
                    "action_scale": motion.get("action_scale"),
                    "controller": motion.get("controller"),
                    "target": motion.get("target"),
                },
                now_sec=time.monotonic(),
            )
            self._last_locomotion_sim_time[robot_id] = current_sim_time

        if not commands:
            return

        message = {
            "message_id": f"core-control-{time.monotonic_ns()}",
            "timestamp": current_sim_time,
            "topic": "locomotion/motion_command",
            "source_module": "core",
            "payload": {
                "commands": commands,
            },
        }
        self.motion_command_pub.publish(String(data=json.dumps(message)))

    def _maybe_start_perception(self) -> None:
        payload = self._perception_payload()
        current_step = int(payload.get("step", 0))
        current_timestamp = float(payload.get("timestamp", 0.0))
        with self._perception_lock:
            cached = self._perception_cache
            last_step = -1 if cached is None else int(cached.get("step", -1))
            last_timestamp = float("-inf") if cached is None else float(cached.get("timestamp", float("-inf")))
            inflight_step = -1 if cached is None else int(cached.get("inflight_step", -1))
            if self._perception_inflight or current_step <= inflight_step:
                return
            if current_step <= last_step:
                return
            if current_timestamp - last_timestamp < self.perception_period:
                return
            self._perception_inflight = True
            if cached is None:
                self._perception_cache = {"inflight_step": current_step}
            else:
                cached["inflight_step"] = current_step
        thread = threading.Thread(target=self._perception_worker, args=(payload,), daemon=True)
        thread.start()

    def _perception_worker(self, payload: dict[str, Any]) -> None:
        record_stem = self._record_perception_request(payload)
        try:
            result = self._post_json(f"{self.perception_url}/estimate", payload, timeout=self.perception_timeout)
            self._record_perception_result(record_stem, result)
            now = time.monotonic()
            step = int(result.get("step", payload.get("step", 0)))
            timestamp = float(result.get("timestamp", payload.get("timestamp", 0.0)))
            with self._perception_lock:
                self._perception_cache = {
                    "result": result,
                    "updated_at": now,
                    "step": step,
                    "timestamp": timestamp,
                    "inflight_step": step,
                }
            self.state_mirror.update_perception(result, now)
        except (TimeoutError, HTTPError, URLError, json.JSONDecodeError, OSError, ValueError) as exc:
            self._record_perception_error(record_stem, exc)
            self._warn_control_once(f"Perception call failed: {exc}")
        finally:
            with self._perception_lock:
                self._perception_inflight = False

    def _latest_perception(self) -> dict[str, Any] | None:
        with self._perception_lock:
            cached = self._perception_cache
            if cached is None:
                return None
            result = cached.get("result")
        return result if isinstance(result, dict) else None

    def _record_perception_request(self, payload: dict[str, Any]) -> Path | None:
        if self.perception_record_dir is None:
            return None
        step = int(payload.get("step", 0))
        timestamp = float(payload.get("timestamp", 0.0))
        stem = f"perception_step_{step:07d}_t_{timestamp:010.3f}_{time.monotonic_ns()}"
        stem_path = self.perception_record_dir / stem
        request_path = stem_path.with_suffix(".request.json")
        body = {
            "record_type": "perception_request",
            "recorded_wall_time": time.time(),
            "payload": payload,
        }
        self._write_record_json(request_path, body)
        return stem_path

    def _record_perception_result(self, stem_path: Path | None, result: dict[str, Any]) -> None:
        if stem_path is None:
            return
        result_path = stem_path.with_suffix(".result.json")
        body = {
            "record_type": "perception_result",
            "recorded_wall_time": time.time(),
            "result": result,
        }
        self._write_record_json(result_path, body)

    def _record_perception_error(self, stem_path: Path | None, exc: Exception) -> None:
        if stem_path is None:
            return
        error_path = stem_path.with_suffix(".error.json")
        body = {
            "record_type": "perception_error",
            "recorded_wall_time": time.time(),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        self._write_record_json(error_path, body)

    def _write_record_json(self, path: Path, body: dict[str, Any]) -> None:
        with self._perception_record_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")

    def _maybe_start_marl(self) -> None:
        payload = self._marl_payload()
        if payload is None:
            return
        current_timestamp = float(payload.get("timestamp", 0.0))
        with self._marl_lock:
            cached = self._marl_cache
            last_timestamp = float("-inf") if cached is None else float(cached.get("timestamp", float("-inf")))
            inflight_timestamp = float("-inf") if cached is None else float(cached.get("inflight_timestamp", float("-inf")))
            if self._marl_inflight or current_timestamp <= inflight_timestamp:
                return
            if current_timestamp - last_timestamp < self.marl_period:
                return
            self._marl_inflight = True
            if cached is None:
                self._marl_cache = {"inflight_timestamp": current_timestamp}
            else:
                cached["inflight_timestamp"] = current_timestamp
        thread = threading.Thread(target=self._marl_worker, args=(payload,), daemon=True)
        thread.start()

    def _marl_worker(self, payload: dict[str, Any]) -> None:
        try:
            result = self._post_json(f"{self.marl_url}/act", payload, timeout=self.marl_timeout)
            now = time.monotonic()
            timestamp = float(result.get("timestamp", payload.get("timestamp", 0.0)))
            merged = {
                **result,
                "input": payload,
            }
            with self._marl_lock:
                self._marl_cache = {
                    "result": merged,
                    "updated_at": now,
                    "timestamp": timestamp,
                    "inflight_timestamp": timestamp,
                }
            self.state_mirror.update_marl(merged, now)
        except (TimeoutError, HTTPError, URLError, json.JSONDecodeError, OSError, ValueError) as exc:
            self._warn_control_once(f"MARL call failed: {exc}")
        finally:
            with self._marl_lock:
                self._marl_inflight = False

    def _latest_marl(self) -> dict[str, Any] | None:
        with self._marl_lock:
            cached = self._marl_cache
            if cached is None:
                return None
            result = cached.get("result")
        return result if isinstance(result, dict) else None

    @staticmethod
    def _marl_subgoal_xy(marl_result: dict[str, Any] | None, robot_id: str) -> list[float] | None:
        if not isinstance(marl_result, dict):
            return None
        subgoals = marl_result.get("subgoals")
        if not isinstance(subgoals, dict):
            return None
        robot = subgoals.get(robot_id)
        if not isinstance(robot, dict):
            return None
        subgoal = robot.get("subgoal")
        if not isinstance(subgoal, list | tuple) or len(subgoal) < 2:
            return None
        return [round(float(subgoal[0]), 4), round(float(subgoal[1]), 4)]

    def _marl_payload(self) -> dict[str, Any] | None:
        perception = self._latest_perception()
        intruder_id = self.intruder_ids[0] if self.intruder_ids else None
        intruder_pose = self.state.intruders.get(intruder_id).pose if intruder_id else None
        intruder_xy = self._perception_intruder_xy(perception) if self.use_perception_output else None
        if intruder_xy is None:
            intruder_xy = self._pose_to_xy(intruder_pose)
        if intruder_xy is None:
            return None
        intruder_velocity = self._intruder_velocity_xy()

        robots: dict[str, Any] = {}
        for robot_id, entity in self.state.robots.items():
            robot_state = self._perception_robot_state(robot_id, perception) if self.use_perception_output else None
            if robot_state is None:
                robot_xy = self._pose_to_xy(entity.pose)
                if robot_xy is None:
                    return None
                velocity = self._robot_velocity_world_xy(robot_id, entity.pose)
                robot_state = {
                    "position": robot_xy,
                    "velocity": velocity,
                }
            robots[robot_id] = robot_state

        sim_state = self.state.latest_simulation_state or {}
        sim_step = int(sim_state.get("timestamp", 0))
        sim_time = sim_step * self.simulation_dt
        self._log_marl_velocity_debug(robots, sim_time)
        return {
            "timestamp": sim_time,
            "robots": robots,
            "intruder": {
                "position": intruder_xy,
                "velocity": intruder_velocity,
            },
        }

    def _robot_velocity_world_xy(self, robot_id: str, pose: PoseStamped | None) -> list[float]:
        base_vel = self._robot_base_velocity_body_xy(robot_id)
        if isinstance(base_vel, list | tuple) and len(base_vel) >= 2:
            yaw = self._pose_yaw(pose)
            vx_b = float(base_vel[0])
            vy_b = float(base_vel[1])
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            vx_w = cos_yaw * vx_b - sin_yaw * vy_b
            vy_w = sin_yaw * vx_b + cos_yaw * vy_b
            return [round(vx_w, 4), round(vy_w, 4)]
        return [0.0, 0.0]

    def _robot_motion_hint_world(self, robot_id: str, pose: PoseStamped | None) -> dict[str, list[float]]:
        linear_xy = self._robot_velocity_world_xy(robot_id, pose)
        linear_velocity = [float(linear_xy[0]), float(linear_xy[1]), 0.0]

        base_ang = self._robot_base_angular_velocity_body(robot_id)
        yaw_rate = float(base_ang[2]) if isinstance(base_ang, list | tuple) and len(base_ang) >= 3 else 0.0
        angular_velocity = [0.0, 0.0, yaw_rate]
        return {
            "linear_velocity_world": linear_velocity,
            "angular_velocity_world": angular_velocity,
        }

    def _robot_base_velocity_body_xy(self, robot_id: str) -> list[float] | None:
        mirror_observation = self.state_mirror.robot_observation_snapshot(robot_id)
        mirror_base_vel = mirror_observation.get("base_linear_velocity")
        if isinstance(mirror_base_vel, list | tuple) and len(mirror_base_vel) >= 2:
            return [float(mirror_base_vel[0]), float(mirror_base_vel[1])]

        observation = self.state.sensors.get(robot_id).locomotion_observation if robot_id in self.state.sensors else None
        raw_base_vel = observation.get("base_linear_velocity") if isinstance(observation, dict) else None
        if isinstance(raw_base_vel, list | tuple) and len(raw_base_vel) >= 2:
            return [float(raw_base_vel[0]), float(raw_base_vel[1])]

        return None

    def _robot_base_angular_velocity_body(self, robot_id: str) -> list[float] | None:
        mirror_observation = self.state_mirror.robot_observation_snapshot(robot_id)
        mirror_base_ang = mirror_observation.get("base_angular_velocity")
        if isinstance(mirror_base_ang, list | tuple) and len(mirror_base_ang) >= 3:
            return [float(mirror_base_ang[0]), float(mirror_base_ang[1]), float(mirror_base_ang[2])]

        observation = self.state.sensors.get(robot_id).locomotion_observation if robot_id in self.state.sensors else None
        raw_base_ang = observation.get("base_angular_velocity") if isinstance(observation, dict) else None
        if isinstance(raw_base_ang, list | tuple) and len(raw_base_ang) >= 3:
            return [float(raw_base_ang[0]), float(raw_base_ang[1]), float(raw_base_ang[2])]

        return None

    def _log_marl_velocity_debug(self, robots: dict[str, Any], sim_time: float) -> None:
        now = time.monotonic()
        if now - self._last_marl_debug_log < 1.0:
            return
        self._last_marl_debug_log = now

        parts: list[str] = []
        for robot_id in self.robot_ids:
            observation = self.state.sensors.get(robot_id).locomotion_observation if robot_id in self.state.sensors else None
            base_vel = observation.get("base_linear_velocity") if isinstance(observation, dict) else None
            planar_speed = observation.get("planar_speed") if isinstance(observation, dict) else None
            mirror_observation = self.state_mirror.robot_observation_snapshot(robot_id)
            mirror_base_vel = mirror_observation.get("base_linear_velocity")
            mirror_planar_speed = mirror_observation.get("planar_speed")
            world_vel = robots.get(robot_id, {}).get("velocity")
            used_base_vel = self._robot_base_velocity_body_xy(robot_id)
            parts.append(
                f"{robot_id}: raw_base_vel_b={base_vel if isinstance(base_vel, list | tuple) else None} "
                f"raw_planar_speed={planar_speed} "
                f"mirror_base_vel_b={mirror_base_vel if isinstance(mirror_base_vel, list | tuple) else None} "
                f"mirror_planar_speed={mirror_planar_speed} used_base_vel_b={used_base_vel} world_vel={world_vel}"
            )
        self.get_logger().info(f"[marl-debug t={sim_time:.3f}] " + " | ".join(parts))

    def _intruder_velocity_xy(self) -> list[float]:
        sim_state = self.state.latest_simulation_state or {}
        intruders = sim_state.get("intruders")
        if isinstance(intruders, dict):
            intruder = intruders.get(self.intruder_ids[0]) if self.intruder_ids else None
            velocity = intruder.get("velocity") if isinstance(intruder, dict) else None
            if isinstance(velocity, list | tuple) and len(velocity) >= 2:
                return [round(float(velocity[0]), 4), round(float(velocity[1]), 4)]
        return [0.0, 0.0]

    def _perception_intruder_xy(self, perception: dict[str, Any] | None) -> list[float] | None:
        if not isinstance(perception, dict):
            return None
        intruder = perception.get("intruder_estimate")
        if not isinstance(intruder, dict) or not intruder.get("detected"):
            return None
        position = intruder.get("position_world")
        if not isinstance(position, list | tuple) or len(position) < 2:
            return None
        return [round(float(position[0]), 3), round(float(position[1]), 3)]

    def _perception_robot_state(self, robot_id: str, perception: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(perception, dict):
            return None
        dogs = perception.get("dogs")
        if not isinstance(dogs, dict):
            return None
        robot = dogs.get(robot_id)
        if not isinstance(robot, dict) or not robot.get("localized"):
            return None
        position = robot.get("position_world")
        velocity = robot.get("velocity_world")
        if not isinstance(position, list | tuple) or len(position) < 2:
            return None
        if not isinstance(velocity, list | tuple) or len(velocity) < 2:
            velocity = [0.0, 0.0]
        return {
            "position": [round(float(position[0]), 3), round(float(position[1]), 3)],
            "velocity": [round(float(velocity[0]), 4), round(float(velocity[1]), 4)],
        }

    def _maybe_start_planning(
        self,
        robot_id: str,
        robot_state: dict[str, Any],
        subgoal: dict[str, Any],
        sensor_payload: dict[str, Any],
        local_goal: list[float],
        robot_yaw: float,
        current_sim_time: float,
    ) -> None:
        with self._plan_lock:
            cached = self._plan_cache.get(robot_id)
            last_planned_at = float("-inf") if cached is None else float(cached.get("sim_timestamp", float("-inf")))
            inflight_sim_time = float("-inf") if cached is None else float(cached.get("inflight_sim_timestamp", float("-inf")))
            if robot_id in self._planning_inflight or current_sim_time <= inflight_sim_time:
                return
            if current_sim_time - last_planned_at < self.planning_period:
                return
            self._planning_inflight.add(robot_id)
            if cached is None:
                self._plan_cache[robot_id] = {"inflight_sim_timestamp": current_sim_time}
            else:
                cached["inflight_sim_timestamp"] = current_sim_time

        payload = {
            "robot_id": robot_id,
            "robot_state": robot_state,
            "subgoal": subgoal["subgoal"],
            "local_goal": local_goal,
            "robot_yaw": robot_yaw,
            "decision": subgoal,
            "sensors": sensor_payload,
            "simulation_state": self.state.latest_simulation_state or {},
            "timestamp": current_sim_time,
        }
        thread = threading.Thread(target=self._planning_worker, args=(robot_id, payload), daemon=True)
        thread.start()

    def _planning_worker(self, robot_id: str, payload: dict[str, Any]) -> None:
        try:
            path = self._post_json(f"{self.navdp_url}/plan", payload, timeout=self.navdp_timeout)
            now = time.monotonic()
            with self._plan_lock:
                self._plan_cache[robot_id] = {
                    "path": path,
                    "updated_at": now,
                    "sim_timestamp": float(payload.get("timestamp", 0.0)),
                    "inflight_sim_timestamp": float(payload.get("timestamp", 0.0)),
                }
            self.state_mirror.update_planning_output(
                robot_id,
                planning={
                    "subgoal": payload.get("decision", {}).get("subgoal"),
                    "local_goal": payload.get("local_goal"),
                    "planner": path.get("planner"),
                    "waypoints": path.get("waypoints", []),
                    "local_waypoints": path.get("local_waypoints", []),
                    "waypoint_count": len(path.get("waypoints", [])) if isinstance(path.get("waypoints"), list) else 0,
                },
                now_sec=now,
            )
        except (TimeoutError, HTTPError, URLError, json.JSONDecodeError, OSError, ValueError) as exc:
            self._warn_control_once(f"NavDP planning failed for {robot_id}: {exc}")
        finally:
            with self._plan_lock:
                self._planning_inflight.discard(robot_id)

    def _latest_path(self, robot_id: str, current_sim_time: float) -> dict[str, Any] | None:
        with self._plan_lock:
            cached = self._plan_cache.get(robot_id)
            if cached is None:
                return None
            if current_sim_time - float(cached.get("sim_timestamp", float("-inf"))) > self.path_stale_after:
                return None
            path = cached.get("path")
        return path if isinstance(path, dict) else None

    def _post_json(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urlrequest.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlrequest.urlopen(request, timeout=timeout) as response:
            data = response.read().decode("utf-8")
        parsed = json.loads(data)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object from {url}")
        if "error" in parsed:
            raise ValueError(str(parsed["error"]))
        return parsed

    def _warn_control_once(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_control_warning < 2.0:
            return
        self._last_control_warning = now
        self.get_logger().warning(message)

    def _current_sim_clock(self) -> tuple[int, float] | None:
        sim_state = self.state.latest_simulation_state or {}
        if "timestamp" not in sim_state:
            return None
        step = int(sim_state.get("timestamp", 0))
        return step, step * self.simulation_dt

    def _heartbeat(self) -> None:
        robot_summary = {
            robot_id: self._pose_to_xy(entity.pose)
            for robot_id, entity in self.state.robots.items()
        }
        intruder_summary = {
            intruder_id: self._pose_to_xy(entity.pose)
            for intruder_id, entity in self.state.intruders.items()
        }
        sensor_summary = {
            robot_id: {
                "camera": sensor.camera_frame_id is not None,
                "depth": sensor.depth_frame_id is not None,
                "semantic": sensor.semantic_frame_id is not None,
                "imu": sensor.imu_frame_id is not None,
                "lidar_samples": sensor.lidar_sample_count,
                "lidar_points": sensor.lidar_point_cloud_count,
            }
            for robot_id, sensor in self.state.sensors.items()
        }
        self.get_logger().info(
            f"state robots={robot_summary} intruders={intruder_summary} sensors={sensor_summary}"
        )

    @staticmethod
    def _pose_to_xy(msg: PoseStamped | None) -> list[float] | None:
        if msg is None:
            return None
        return [round(msg.pose.position.x, 3), round(msg.pose.position.y, 3)]

    def _sensor_payload(self, robot_id: str) -> dict[str, Any]:
        sensor = self.state.sensors[robot_id]
        payload: dict[str, Any] = {}
        if sensor.camera_payload is not None:
            payload["rgb"] = sensor.camera_payload
        if sensor.depth_payload is not None:
            payload["depth"] = sensor.depth_payload
        if sensor.semantic_payload is not None:
            payload["semantic_segmentation"] = sensor.semantic_payload
        if sensor.imu_payload is not None:
            payload["imu"] = sensor.imu_payload
        if sensor.lidar_payload is not None:
            payload["lidar_scan"] = sensor.lidar_payload
        if sensor.lidar_point_cloud_payload is not None:
            payload["lidar_points"] = sensor.lidar_point_cloud_payload
        return payload

    def _perception_payload(self) -> dict[str, Any]:
        sim_state = self.state.latest_simulation_state or {}
        camera_infos = sim_state.get("camera_infos", {}) if isinstance(sim_state, dict) else {}
        robot_camera_infos = camera_infos.get("robots", {}) if isinstance(camera_infos, dict) else {}
        cctv_camera_infos = camera_infos.get("cctv", {}) if isinstance(camera_infos, dict) else {}

        robots: dict[str, Any] = {}
        for robot_id in self.robot_ids:
            entity = self.state.robots.get(robot_id)
            sensors = self.state.sensors.get(robot_id)
            robots[robot_id] = {
                "pose": self._pose_payload(entity.pose) if entity and entity.pose is not None else None,
                "motion_hint": self._robot_motion_hint_world(robot_id, entity.pose) if entity and entity.pose is not None else None,
                "sensors": {
                    "rgb": self._camera_payload_with_meta(
                        sensors.camera_payload if sensors else None,
                        robot_camera_infos.get(robot_id, {}) if isinstance(robot_camera_infos, dict) else {},
                    ),
                    "depth": sensors.depth_payload if sensors else None,
                    "semantic_segmentation": sensors.semantic_payload if sensors else None,
                    "info": self._camera_semantic_info(
                        robot_camera_infos.get(robot_id, {}) if isinstance(robot_camera_infos, dict) else {}
                    ),
                    "imu": sensors.imu_payload if sensors else None,
                    "lidar_points": sensors.lidar_point_cloud_payload if sensors else None,
                },
            }

        intruders: dict[str, Any] = {}
        for intruder_id in self.intruder_ids:
            entity = self.state.intruders.get(intruder_id)
            intruders[intruder_id] = {
                "pose": self._pose_payload(entity.pose) if entity and entity.pose is not None else None,
            }

        cctv: dict[str, Any] = {}
        for camera_id in self.cctv_ids:
            camera_state = self.state.cctv.get(camera_id)
            camera_meta = cctv_camera_infos.get(camera_id, {}) if isinstance(cctv_camera_infos, dict) else {}
            cctv[camera_id] = {
                "rgb": self._camera_payload_with_meta(
                    camera_state.camera_payload if camera_state else None,
                    camera_meta,
                ),
                "semantic_segmentation": camera_state.semantic_payload if camera_state else None,
                "info": self._camera_semantic_info(camera_meta),
                "pos_w": camera_meta.get("pos_w") if isinstance(camera_meta, dict) else None,
                "quat_w": camera_meta.get("quat_w") if isinstance(camera_meta, dict) else None,
                "intrinsic_matrix": camera_meta.get("intrinsic_matrix") if isinstance(camera_meta, dict) else None,
            }

        sim_step = int(sim_state.get("timestamp", 0))
        sim_time = sim_step * self.simulation_dt

        semantic_labels: dict[str, Any] = {}
        suspect_id = self._suspect_id_from_camera_infos(camera_infos)
        if suspect_id is not None:
            semantic_labels["suspect_id"] = suspect_id

        return {
            "step": sim_step,
            "timestamp": sim_time,
            "robots": robots,
            "intruders": intruders,
            "cctv": cctv,
            "semantic_labels": semantic_labels,
            "simulation_state": sim_state,
        }

    @staticmethod
    def _camera_semantic_info(meta: Any) -> dict[str, Any]:
        if not isinstance(meta, dict):
            return {}
        info = meta.get("info")
        return info if isinstance(info, dict) else {}

    def _camera_payload_with_meta(self, payload: Any, meta: Any) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return payload if payload is None else None
        merged = dict(payload)
        if isinstance(meta, dict):
            for key in ("pos_w", "quat_w", "intrinsic_matrix"):
                value = meta.get(key)
                if value is not None:
                    merged[key] = value
        return merged

    def _suspect_id_from_camera_infos(self, camera_infos: dict[str, Any]) -> int | None:
        if not isinstance(camera_infos, dict):
            return None
        groups = []
        for key in ("cctv", "robots"):
            group = camera_infos.get(key)
            if isinstance(group, dict):
                groups.append(group)
        for group in groups:
            for info in group.values():
                if not isinstance(info, dict):
                    continue
                semantic_info = self._camera_semantic_info(info)
                id_to_labels = semantic_info.get("idToLabels", {})
                if not isinstance(id_to_labels, dict):
                    continue
                for sem_id, label_info in id_to_labels.items():
                    label_str = ""
                    if isinstance(label_info, str):
                        label_str = label_info
                    elif isinstance(label_info, dict):
                        label_str = str(label_info.get("class", ""))
                    if "suspect" in label_str.lower():
                        try:
                            return int(sem_id)
                        except (TypeError, ValueError):
                            continue
        return None

    def _body_command_from_path(
        self,
        pose: PoseStamped | None,
        robot_xy: list[float],
        path: dict[str, Any],
    ) -> list[float]:
        waypoints = path.get("waypoints", [])
        if not isinstance(waypoints, list) or not waypoints:
            return [0.0, 0.0, 0.0]

        target_xy = waypoints[-1]
        for waypoint in waypoints:
            if not isinstance(waypoint, list | tuple) or len(waypoint) < 2:
                continue
            distance = math.hypot(float(waypoint[0]) - float(robot_xy[0]), float(waypoint[1]) - float(robot_xy[1]))
            if distance >= 0.35:
                target_xy = waypoint
                break

        return self._world_point_to_body_velocity(pose, target_xy, robot_xy)

    def _body_command_from_navdp_path(
        self,
        pose: PoseStamped | None,
        robot_xy: list[float],
        path: dict[str, Any],
    ) -> list[float]:
        """Use NavDP's native local trajectory for the low-level velocity command."""
        local_waypoints = path.get("local_waypoints")
        if isinstance(local_waypoints, list) and local_waypoints:
            target = local_waypoints[-1]
            for waypoint in local_waypoints:
                if not isinstance(waypoint, list | tuple) or len(waypoint) < 2:
                    continue
                distance = math.hypot(float(waypoint[0]), float(waypoint[1]))
                if distance >= 0.35:
                    target = waypoint
                    break
            if isinstance(target, list | tuple) and len(target) >= 2:
                return self._local_point_to_body_velocity(target)

        return self._body_command_from_path(pose, robot_xy, path)

    def _local_point_to_body_velocity(self, target_xy: list[float] | tuple[float, float]) -> list[float]:
        x = float(target_xy[0])
        y = float(target_xy[1])
        distance = math.hypot(x, y)
        if distance < 1.0e-6:
            return [0.0, 0.0, 0.0]
        speed = min(0.8, distance)
        return [round(x / distance * speed, 4), round(y / distance * speed, 4), 0.0]

    def _world_point_to_body_velocity(
        self,
        pose: PoseStamped | None,
        target_xy: list[float] | tuple[float, float],
        robot_xy: list[float],
    ) -> list[float]:
        body_xy = self._world_point_to_body_xy(pose, target_xy, robot_xy)
        dx = body_xy[0]
        dy = body_xy[1]
        distance = math.hypot(dx, dy)
        if distance < 1.0e-6:
            return [0.0, 0.0, 0.0]
        speed = min(0.8, distance)
        return [round(dx / distance * speed, 4), round(dy / distance * speed, 4), 0.0]

    def _world_point_to_body_xy(
        self,
        pose: PoseStamped | None,
        target_xy: list[float] | tuple[float, float],
        robot_xy: list[float],
    ) -> list[float]:
        dx = float(target_xy[0]) - float(robot_xy[0])
        dy = float(target_xy[1]) - float(robot_xy[1])
        yaw = self._pose_yaw(pose)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        x_body = cos_yaw * dx + sin_yaw * dy
        y_body = -sin_yaw * dx + cos_yaw * dy
        return [round(x_body, 4), round(y_body, 4)]

    @staticmethod
    def _image_payload(msg: Image) -> dict[str, Any]:
        return {
            "frame_id": msg.header.frame_id,
            "height": int(msg.height),
            "width": int(msg.width),
            "encoding": msg.encoding,
            "is_bigendian": bool(msg.is_bigendian),
            "step": int(msg.step),
            "data_base64": base64.b64encode(bytes(msg.data)).decode("ascii"),
        }

    @staticmethod
    def _point_cloud_payload(msg: PointCloud2) -> dict[str, Any]:
        return {
            "frame_id": msg.header.frame_id,
            "height": int(msg.height),
            "width": int(msg.width),
            "point_step": int(msg.point_step),
            "row_step": int(msg.row_step),
            "is_bigendian": bool(msg.is_bigendian),
            "is_dense": bool(msg.is_dense),
            "fields": [
                {
                    "name": field.name,
                    "offset": int(field.offset),
                    "datatype": int(field.datatype),
                    "count": int(field.count),
                }
                for field in msg.fields
            ],
            "data_base64": base64.b64encode(bytes(msg.data)).decode("ascii"),
        }

    @staticmethod
    def _laser_scan_payload(msg: LaserScan) -> dict[str, Any]:
        return {
            "frame_id": msg.header.frame_id,
            "angle_min": float(msg.angle_min),
            "angle_max": float(msg.angle_max),
            "angle_increment": float(msg.angle_increment),
            "time_increment": float(msg.time_increment),
            "scan_time": float(msg.scan_time),
            "range_min": float(msg.range_min),
            "range_max": float(msg.range_max),
            "ranges": [float(value) for value in msg.ranges],
            "intensities": [float(value) for value in msg.intensities],
        }

    @staticmethod
    def _imu_payload(msg: Imu) -> dict[str, Any]:
        return {
            "frame_id": msg.header.frame_id,
            "orientation": [
                float(msg.orientation.w),
                float(msg.orientation.x),
                float(msg.orientation.y),
                float(msg.orientation.z),
            ],
            "angular_velocity": [
                float(msg.angular_velocity.x),
                float(msg.angular_velocity.y),
                float(msg.angular_velocity.z),
            ],
            "linear_acceleration": [
                float(msg.linear_acceleration.x),
                float(msg.linear_acceleration.y),
                float(msg.linear_acceleration.z),
            ],
        }

    @staticmethod
    def _pose_payload(msg: PoseStamped) -> dict[str, Any]:
        return {
            "frame_id": msg.header.frame_id,
            "position": [
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
            ],
            "orientation": [
                float(msg.pose.orientation.w),
                float(msg.pose.orientation.x),
                float(msg.pose.orientation.y),
                float(msg.pose.orientation.z),
            ],
            "yaw": float(CoreControlNode._pose_yaw(msg)),
        }

    @staticmethod
    def _pose_yaw(msg: PoseStamped | None) -> float:
        if msg is None:
            return 0.0
        q = msg.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _valid_velocity(value: Any) -> bool:
        return (
            isinstance(value, list | tuple)
            and len(value) >= 2
            and all(isinstance(item, int | float) for item in value[:2])
        )

    def destroy_node(self) -> bool:
        if self._server is not None:
            self._server.should_exit = True
        return super().destroy_node()


def create_state_app(state_mirror: StateMirror, websocket_period: float) -> FastAPI:
    app = FastAPI(title="Factory Core State API")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "owner": "core"}

    @app.get("/api/state")
    async def api_state(
        include_images: bool = Query(default=False),
        max_lidar_points: int = Query(default=120, ge=0, le=2000),
    ) -> JSONResponse:
        return JSONResponse(state_mirror.snapshot(include_images=include_images, max_lidar_points=max_lidar_points))

    @app.websocket("/ws")
    async def websocket_state(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(
                    state_mirror.snapshot(include_images=True, max_lidar_points=220)
                )
                await asyncio.sleep(websocket_period)
        except WebSocketDisconnect:
            return

    return app


def main() -> None:
    rclpy.init()
    node = CoreControlNode()
    try:
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
