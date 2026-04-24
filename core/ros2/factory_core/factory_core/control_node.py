from __future__ import annotations

import asyncio
import base64
import json
import math
import threading
import time
from dataclasses import dataclass, field
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
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String


DEFAULT_ROBOTS = ["agent_1", "agent_2"]
DEFAULT_INTRUDERS = ["intruder_1"]


@dataclass
class EntityState:
    pose: PoseStamped | None = None


@dataclass
class SensorState:
    camera_frame_id: str | None = None
    camera_stamp_sec: int | None = None
    camera_msg: Image | None = None
    depth_frame_id: str | None = None
    depth_stamp_sec: int | None = None
    depth_msg: Image | None = None
    locomotion_observation: dict[str, Any] | None = None
    lidar_frame_id: str | None = None
    lidar_stamp_sec: int | None = None
    lidar_sample_count: int = 0


@dataclass
class CoreState:
    robots: dict[str, EntityState] = field(default_factory=dict)
    intruders: dict[str, EntityState] = field(default_factory=dict)
    sensors: dict[str, SensorState] = field(default_factory=dict)
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
        self.declare_parameter("navdp_url", "http://127.0.0.1:8889")
        self.declare_parameter("locomotion_url", "http://127.0.0.1:8890")

        self.robot_ids = list(self.get_parameter("robot_ids").value)
        self.intruder_ids = list(self.get_parameter("intruder_ids").value)
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
        self.navdp_url = str(self.get_parameter("navdp_url").value).rstrip("/")
        self.locomotion_url = str(self.get_parameter("locomotion_url").value).rstrip("/")

        self.state = CoreState(
            robots={robot_id: EntityState() for robot_id in self.robot_ids},
            intruders={intruder_id: EntityState() for intruder_id in self.intruder_ids},
            sensors={robot_id: SensorState() for robot_id in self.robot_ids},
        )
        self.state_mirror = StateMirror(self.robot_ids, self.intruder_ids, stale_after, self.topic_prefix)
        self.motion_command_pub = self.create_publisher(
            String,
            f"{self.control_topic_prefix}/locomotion/motion_command",
            10,
        )
        self._last_control_warning = 0.0
        self._plan_cache: dict[str, dict[str, Any]] = {}
        self._planning_inflight: set[str] = set()
        self._plan_lock = threading.Lock()

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
                LaserScan,
                f"{self.topic_prefix}/{robot_id}/lidar/scan",
                lambda msg, rid=robot_id: self._on_lidar(rid, msg),
                10,
                callback_group=self._state_group,
            )
            self.create_subscription(
                String,
                f"{self.topic_prefix}/{robot_id}/locomotion/observation",
                lambda msg, rid=robot_id: self._on_locomotion_observation(rid, msg),
                20,
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
        self.state_mirror.update_camera(robot_id, msg, time.monotonic())

    def _on_depth(self, robot_id: str, msg: Image) -> None:
        sensor = self.state.sensors[robot_id]
        sensor.depth_frame_id = msg.header.frame_id
        sensor.depth_stamp_sec = int(msg.header.stamp.sec)
        sensor.depth_msg = msg

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
        self.state_mirror.update_lidar(robot_id, msg, time.monotonic())

    def _run_state_server(self) -> None:
        app = create_state_app(self.state_mirror, self.state_websocket_period)
        config = uvicorn.Config(app, host=self.state_host, port=self.state_port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._server.run()

    def _control_tick(self) -> None:
        intruder_id = self.intruder_ids[0] if self.intruder_ids else None
        intruder_pose = self.state.intruders.get(intruder_id).pose if intruder_id else None
        intruder_xy = self._pose_to_xy(intruder_pose)
        if intruder_xy is None:
            return

        commands: dict[str, dict[str, Any]] = {}
        for robot_id, entity in self.state.robots.items():
            robot_xy = self._pose_to_xy(entity.pose)
            if robot_xy is None:
                continue

            robot_state = {"position": robot_xy, "velocity": [0.0, 0.0]}
            sensor_payload = self._sensor_payload(robot_id)
            subgoal = {
                "robot_id": robot_id,
                "subgoal": intruder_xy,
                "mode": "chase",
                "priority": 1,
            }
            robot_yaw = self._pose_yaw(entity.pose)
            local_goal = self._world_point_to_body_xy(entity.pose, intruder_xy, robot_xy)
            self._maybe_start_planning(robot_id, robot_state, subgoal, sensor_payload, local_goal, robot_yaw)
            path = self._latest_path(robot_id)
            if path is None:
                continue
            try:
                body_command = self._body_command_from_navdp_path(entity.pose, robot_xy, path)
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
            self.state_mirror.update_control_outputs(
                robot_id,
                planning={
                    "subgoal": subgoal["subgoal"],
                    "local_goal": local_goal,
                    "planner": path.get("planner"),
                    "waypoints": path.get("waypoints", []),
                    "local_waypoints": path.get("local_waypoints", []),
                    "waypoint_count": len(path.get("waypoints", [])) if isinstance(path.get("waypoints"), list) else 0,
                },
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

        if not commands:
            return

        message = {
            "message_id": f"core-control-{time.monotonic_ns()}",
            "timestamp": time.monotonic(),
            "topic": "locomotion/motion_command",
            "source_module": "core",
            "payload": {
                "commands": commands,
            },
        }
        self.motion_command_pub.publish(String(data=json.dumps(message)))

    def _maybe_start_planning(
        self,
        robot_id: str,
        robot_state: dict[str, Any],
        subgoal: dict[str, Any],
        sensor_payload: dict[str, Any],
        local_goal: list[float],
        robot_yaw: float,
    ) -> None:
        now = time.monotonic()
        with self._plan_lock:
            cached = self._plan_cache.get(robot_id)
            last_planned_at = 0.0 if cached is None else float(cached.get("updated_at", 0.0))
            if robot_id in self._planning_inflight or now - last_planned_at < self.planning_period:
                return
            self._planning_inflight.add(robot_id)

        payload = {
            "robot_id": robot_id,
            "robot_state": robot_state,
            "subgoal": subgoal["subgoal"],
            "local_goal": local_goal,
            "robot_yaw": robot_yaw,
            "decision": subgoal,
            "sensors": sensor_payload,
            "simulation_state": self.state.latest_simulation_state or {},
        }
        thread = threading.Thread(target=self._planning_worker, args=(robot_id, payload), daemon=True)
        thread.start()

    def _planning_worker(self, robot_id: str, payload: dict[str, Any]) -> None:
        try:
            path = self._post_json(f"{self.navdp_url}/plan", payload, timeout=self.navdp_timeout)
            with self._plan_lock:
                self._plan_cache[robot_id] = {"path": path, "updated_at": time.monotonic()}
        except (TimeoutError, HTTPError, URLError, json.JSONDecodeError, OSError, ValueError) as exc:
            self._warn_control_once(f"NavDP planning failed for {robot_id}: {exc}")
        finally:
            with self._plan_lock:
                self._planning_inflight.discard(robot_id)

    def _latest_path(self, robot_id: str) -> dict[str, Any] | None:
        now = time.monotonic()
        with self._plan_lock:
            cached = self._plan_cache.get(robot_id)
            if cached is None:
                return None
            if now - float(cached.get("updated_at", 0.0)) > self.path_stale_after:
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
                "lidar_samples": sensor.lidar_sample_count,
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
        if sensor.camera_msg is not None:
            payload["rgb"] = self._image_payload(sensor.camera_msg)
        if sensor.depth_msg is not None:
            payload["depth"] = self._image_payload(sensor.depth_msg)
        return payload

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
