from __future__ import annotations

import json
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String


DEFAULT_ROBOTS = ["agent_1", "agent_2"]
DEFAULT_INTRUDERS = ["intruder_1"]


class MockSimulationPublisher(Node):
    """Publishes the ROS2 topic contract expected from simulation.

    This is a temporary bridge for integration testing. The Isaac Sim bridge
    should publish the same topics from real camera, LiDAR, and actor states.
    """

    def __init__(self) -> None:
        super().__init__("factory_mock_sim_publisher")

        self.declare_parameter("robot_ids", DEFAULT_ROBOTS)
        self.declare_parameter("intruder_ids", DEFAULT_INTRUDERS)
        self.declare_parameter("topic_prefix", "/factory/simulation")
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("image_width", 64)
        self.declare_parameter("image_height", 48)

        self.robot_ids = list(self.get_parameter("robot_ids").value)
        self.intruder_ids = list(self.get_parameter("intruder_ids").value)
        self.topic_prefix = str(self.get_parameter("topic_prefix").value).rstrip("/")
        publish_rate = float(self.get_parameter("publish_rate").value)
        self.image_width = int(self.get_parameter("image_width").value)
        self.image_height = int(self.get_parameter("image_height").value)

        self.state_pub = self.create_publisher(String, f"{self.topic_prefix}/state", 10)
        self.robot_pose_pubs = {
            robot_id: self.create_publisher(PoseStamped, f"{self.topic_prefix}/{robot_id}/pose", 20)
            for robot_id in self.robot_ids
        }
        self.intruder_pose_pubs = {
            intruder_id: self.create_publisher(PoseStamped, f"{self.topic_prefix}/{intruder_id}/pose", 20)
            for intruder_id in self.intruder_ids
        }
        self.camera_pubs = {
            robot_id: self.create_publisher(Image, f"{self.topic_prefix}/{robot_id}/camera/image_raw", 5)
            for robot_id in self.robot_ids
        }
        self.lidar_pubs = {
            robot_id: self.create_publisher(LaserScan, f"{self.topic_prefix}/{robot_id}/lidar/scan", 10)
            for robot_id in self.robot_ids
        }

        self.step_idx = 0
        self.create_timer(1.0 / publish_rate, self._publish_frame)
        self.get_logger().info(
            f"Mock simulation publisher publishing under {self.topic_prefix} at {publish_rate} Hz"
        )

    def _publish_frame(self) -> None:
        stamp = self.get_clock().now().to_msg()
        t = self.step_idx * 0.1

        robot_states = {}
        for index, robot_id in enumerate(self.robot_ids):
            x = -2.0 + 0.15 * math.sin(t + index)
            y = -2.0 + index * 3.6 + 0.15 * math.cos(t + index)
            robot_states[robot_id] = (x, y, 0.42)
            self.robot_pose_pubs[robot_id].publish(self._make_pose(robot_id, x, y, 0.42, stamp))
            self.camera_pubs[robot_id].publish(self._make_image(robot_id, stamp, self.step_idx + index * 32))
            self.lidar_pubs[robot_id].publish(self._make_scan(robot_id, stamp, t + index))

        intruder_states = {}
        for index, intruder_id in enumerate(self.intruder_ids):
            x = 2.0 + 0.1 * math.sin(t * 0.5 + index)
            y = -0.5 + 0.1 * math.cos(t * 0.5 + index)
            intruder_states[intruder_id] = (x, y, 1.05)
            self.intruder_pose_pubs[intruder_id].publish(self._make_pose(intruder_id, x, y, 1.05, stamp))

        self.state_pub.publish(String(data=json.dumps(self._make_state(robot_states, intruder_states, t))))
        self.step_idx += 1

    @staticmethod
    def _make_pose(entity_id: str, x: float, y: float, z: float, stamp) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "world"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = 1.0
        return msg

    def _make_image(self, robot_id: str, stamp, phase: int) -> Image:
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = f"{robot_id}/front_camera"
        msg.height = self.image_height
        msg.width = self.image_width
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = self.image_width * 3

        data = bytearray(self.image_height * msg.step)
        for row in range(self.image_height):
            for col in range(self.image_width):
                offset = row * msg.step + col * 3
                data[offset] = (col + phase) % 256
                data[offset + 1] = (row * 2 + phase) % 256
                data[offset + 2] = 120
        msg.data = bytes(data)
        return msg

    @staticmethod
    def _make_scan(robot_id: str, stamp, phase: float) -> LaserScan:
        msg = LaserScan()
        msg.header.stamp = stamp
        msg.header.frame_id = f"{robot_id}/front_lidar"
        msg.angle_min = -math.pi
        msg.angle_max = math.pi
        msg.angle_increment = math.radians(1.0)
        msg.time_increment = 0.0
        msg.scan_time = 0.1
        msg.range_min = 0.05
        msg.range_max = 30.0

        count = int((msg.angle_max - msg.angle_min) / msg.angle_increment) + 1
        msg.ranges = [
            4.0 + 0.5 * math.sin(phase + idx * 0.05)
            for idx in range(count)
        ]
        msg.intensities = [1.0 for _ in range(count)]
        return msg

    @staticmethod
    def _make_state(
        robot_states: dict[str, tuple[float, float, float]],
        intruder_states: dict[str, tuple[float, float, float]],
        timestamp: float,
    ) -> dict:
        return {
            "timestamp": timestamp,
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


def main() -> None:
    rclpy.init()
    node = MockSimulationPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
