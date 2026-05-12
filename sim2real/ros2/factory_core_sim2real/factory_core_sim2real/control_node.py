from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import rclpy
import uvicorn
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, PointCloud2

from factory_core_sim2real.state_mirror import StateMirror

try:
    from unitree_go.msg import SportModeState
except Exception:  # pragma: no cover - depends on local ROS message workspace
    SportModeState = None

DEFAULT_ROBOT_ID = "agent_1"


@dataclass
class RobotRuntimeState:
    pose_xy: list[float] | None = None
    velocity_xy: list[float] | None = None
    latest_sport: dict[str, Any] | None = None


class CoreControlNode(Node):
    """Minimal sim2real Core entrypoint for the first batch of true-robot topics.

    This node intentionally focuses on:
    - /sportmodestate
    - /utlidar/robot_pose
    - /utlidar/robot_odom

    It mirrors the incoming robot state into the existing state API / websocket
    contract so the dashboard can render the robot without the rest of the
    simulation-era module stack.
    """

    def __init__(self) -> None:
        super().__init__("factory_core_sim2real_control")

        self.declare_parameter("robot_id", DEFAULT_ROBOT_ID)
        self.declare_parameter("topic_prefix", "/factory/sim2real")
        self.declare_parameter("stale_after", 1.0)
        self.declare_parameter("state_host", "0.0.0.0")
        self.declare_parameter("state_port", 8765)
        self.declare_parameter("state_websocket_period", 0.1)
        self.declare_parameter("sport_mode_topic", "/sportmodestate")
        self.declare_parameter("robot_pose_topic", "/utlidar/robot_pose")
        self.declare_parameter("robot_odom_topic", "/utlidar/robot_odom")
        self.declare_parameter("imu_topic", "/utlidar/imu")
        self.declare_parameter("lidar_points_topic", "/utlidar/cloud")
        self.declare_parameter("camera_topic", "/factory/sim2real/agent_1/camera/image_raw")
        self.declare_parameter("camera_cache_path", "/tmp/factory_sim2real/front_camera.jpg")
        self.declare_parameter("camera_frame_id", "front_camera")

        self.robot_id = str(self.get_parameter("robot_id").value)
        self.robot_ids = [self.robot_id]
        self.topic_prefix = str(self.get_parameter("topic_prefix").value).rstrip("/")
        self.state_host = str(self.get_parameter("state_host").value)
        self.state_port = int(self.get_parameter("state_port").value)
        self.state_websocket_period = float(self.get_parameter("state_websocket_period").value)
        self.sport_mode_topic = str(self.get_parameter("sport_mode_topic").value).strip()
        self.robot_pose_topic = str(self.get_parameter("robot_pose_topic").value).strip()
        self.robot_odom_topic = str(self.get_parameter("robot_odom_topic").value).strip()
        self.imu_topic = str(self.get_parameter("imu_topic").value).strip()
        self.lidar_points_topic = str(self.get_parameter("lidar_points_topic").value).strip()
        self.camera_topic = str(self.get_parameter("camera_topic").value).strip()
        self.camera_cache_path = str(self.get_parameter("camera_cache_path").value).strip()
        self.camera_frame_id = str(self.get_parameter("camera_frame_id").value).strip()
        self._camera_cache_mtime_ns: int | None = None

        self.runtime = RobotRuntimeState()
        self.state_mirror = StateMirror(
            robot_ids=self.robot_ids,
            intruder_ids=[],
            stale_after=float(self.get_parameter("stale_after").value),
            topic_prefix=self.topic_prefix,
            cctv_ids=[],
        )

        self._server: uvicorn.Server | None = None
        self._server_thread = threading.Thread(target=self._run_state_server, daemon=True)
        self._server_thread.start()

        self.create_subscription(PoseStamped, self.robot_pose_topic, self._on_robot_pose, 10)
        self.create_subscription(Odometry, self.robot_odom_topic, self._on_robot_odom, 10)
        if self.camera_topic:
            self.create_subscription(Image, self.camera_topic, self._on_camera, 10)
        if self.imu_topic:
            self.create_subscription(Imu, self.imu_topic, self._on_imu, 10)
        if self.lidar_points_topic:
            self.create_subscription(PointCloud2, self.lidar_points_topic, self._on_lidar_points, 10)

        if self.sport_mode_topic and SportModeState is not None:
            self.create_subscription(SportModeState, self.sport_mode_topic, self._on_sport_mode_state, 10)
        elif self.sport_mode_topic:
            self.get_logger().warning(
            "unitree_go.msg.SportModeState is unavailable; /sportmodestate will not be consumed"
            )

        self.create_timer(0.2, self._publish_aggregate_snapshot)
        if self.camera_cache_path:
            self.create_timer(0.15, self._poll_camera_cache)
        self.create_timer(1.0, self._log_status)
        self.get_logger().info(
            "Sim2real core listening under "
            f"{self.topic_prefix} for robots={self.robot_ids}, "
            f"sport={self.sport_mode_topic}, pose={self.robot_pose_topic}, odom={self.robot_odom_topic}, "
            f"camera={self.camera_topic or 'disabled'}, camera_cache={self.camera_cache_path or 'disabled'}, "
            f"imu={self.imu_topic}, lidar_points={self.lidar_points_topic}"
        )

    def _on_robot_pose(self, msg: PoseStamped) -> None:
        self.runtime.pose_xy = [float(msg.pose.position.x), float(msg.pose.position.y)]
        self.state_mirror.update_robot_pose(self.robot_id, msg, time.monotonic())

    def _on_robot_odom(self, msg: Odometry) -> None:
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        self.runtime.pose_xy = [float(msg.pose.pose.position.x), float(msg.pose.pose.position.y)]
        self.runtime.velocity_xy = [float(msg.twist.twist.linear.x), float(msg.twist.twist.linear.y)]
        self.state_mirror.update_robot_pose(self.robot_id, pose, time.monotonic())

    def _on_sport_mode_state(self, msg: Any) -> None:
        payload = {
            "mode": int(msg.mode),
            "gait_type": int(msg.gait_type),
            "progress": float(msg.progress),
            "body_height": float(msg.body_height),
            "foot_raise_height": float(msg.foot_raise_height),
            "position": [float(v) for v in list(msg.position[:3])],
            "velocity": [float(v) for v in list(msg.velocity[:3])],
            "yaw_speed": float(msg.yaw_speed),
            "range_obstacle": [float(v) for v in list(msg.range_obstacle)],
            "foot_force": [float(v) for v in list(msg.foot_force)],
            "imu": {
                "quaternion": [float(v) for v in list(msg.imu_state.quaternion[:4])],
                "gyroscope": [float(v) for v in list(msg.imu_state.gyroscope[:3])],
                "accelerometer": [float(v) for v in list(msg.imu_state.accelerometer[:3])],
            },
        }
        self.runtime.latest_sport = payload
        self.state_mirror.update_robot_status(self.robot_id, payload, time.monotonic())

    def _on_camera(self, msg: Image) -> None:
        self.state_mirror.update_camera(self.robot_id, msg, time.monotonic())

    def _poll_camera_cache(self) -> None:
        if not self.camera_cache_path:
            return
        try:
            stat = os.stat(self.camera_cache_path)
        except FileNotFoundError:
            return
        except Exception as exc:
            self.get_logger().warning(f"Camera cache stat failed: {exc}")
            return

        if self._camera_cache_mtime_ns == stat.st_mtime_ns:
            return

        try:
            with open(self.camera_cache_path, "rb") as handle:
                jpeg_bytes = handle.read()
        except Exception as exc:
            self.get_logger().warning(f"Camera cache read failed: {exc}")
            return

        if not jpeg_bytes:
            return

        self._camera_cache_mtime_ns = stat.st_mtime_ns
        self.state_mirror.update_camera_jpeg(
            self.robot_id,
            jpeg_bytes,
            self.camera_frame_id,
            time.monotonic(),
        )

    def _on_imu(self, msg: Imu) -> None:
        self.state_mirror.update_imu(self.robot_id, msg, time.monotonic())

    def _on_lidar_points(self, msg: PointCloud2) -> None:
        self.state_mirror.update_lidar_points(self.robot_id, msg, time.monotonic())

    def _publish_aggregate_snapshot(self) -> None:
        payload = {
            "timestamp": time.time(),
            "frame_id": "world",
            "robots": {self.robot_id: self.runtime.pose_xy},
            "robot_velocities": {self.robot_id: self.runtime.velocity_xy},
            "intruders": {},
            "source_module": "sim2real_core",
        }
        self.state_mirror.update_aggregate(json.dumps(payload), time.monotonic())

    def _log_status(self) -> None:
        robot = self.state_mirror.snapshot(include_images=False)["robots"][self.robot_id]
        self.get_logger().info(
            f"robot={self.robot_id} "
            f"pose={robot['pose']['fresh']} "
            f"status={robot['status']['fresh']} "
            f"camera={robot['camera']['fresh']} "
            f"imu={robot['imu']['fresh']} "
            f"lidar_points={robot['lidar_points']['fresh']}"
        )

    def _run_state_server(self) -> None:
        app = create_state_app(self.state_mirror, self.state_websocket_period)
        self.get_logger().info(f"Sim2real state API listening on http://{self.state_host}:{self.state_port}")
        config = uvicorn.Config(app, host=self.state_host, port=self.state_port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._server.run()

    def destroy_node(self) -> bool:
        if self._server is not None:
            self._server.should_exit = True
        return super().destroy_node()


def create_state_app(state_mirror: StateMirror, websocket_period: float) -> FastAPI:
    app = FastAPI(title="Factory Sim2Real Core")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "owner": "core_sim2real"}

    @app.get("/api/state")
    async def api_state(
        include_images: bool = Query(True),
        max_lidar_points: int = Query(0, ge=0, le=5000),
    ) -> JSONResponse:
        return JSONResponse(state_mirror.snapshot(include_images=include_images, max_lidar_points=max_lidar_points))

    @app.websocket("/ws")
    async def websocket_state(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(
                    state_mirror.snapshot(include_images=True, max_lidar_points=0)
                )
                await asyncio.sleep(websocket_period)
        except WebSocketDisconnect:
            return

    return app


def main() -> None:
    rclpy.init()
    executor: MultiThreadedExecutor | None = None
    node = CoreControlNode()
    try:
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    finally:
        if executor is not None:
            executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
