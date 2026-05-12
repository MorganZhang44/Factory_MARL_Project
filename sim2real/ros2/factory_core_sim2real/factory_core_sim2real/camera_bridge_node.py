from __future__ import annotations

import io
import threading
import time

import rclpy
from PIL import Image as PillowImage
from rclpy.node import Node
from sensor_msgs.msg import Image
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient


class FrontCameraBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("factory_core_sim2real_camera_bridge")

        self.declare_parameter("camera_interface", "eno1")
        self.declare_parameter("camera_topic", "/factory/sim2real/agent_1/camera/image_raw")
        self.declare_parameter("camera_poll_hz", 8.0)
        self.declare_parameter("camera_frame_id", "front_camera")

        self.camera_interface = str(self.get_parameter("camera_interface").value).strip()
        self.camera_topic = str(self.get_parameter("camera_topic").value).strip()
        self.camera_poll_hz = float(self.get_parameter("camera_poll_hz").value)
        self.camera_frame_id = str(self.get_parameter("camera_frame_id").value).strip()

        self.publisher = self.create_publisher(Image, self.camera_topic, 10)
        self._client: VideoClient | None = None
        self._stop = threading.Event()

        try:
            ChannelFactoryInitialize(0, self.camera_interface or None)
            self._client = VideoClient()
            self._client.SetTimeout(3.0)
            self._client.Init()
            self._thread = threading.Thread(target=self._camera_loop, daemon=True, name="go2_front_camera_bridge")
            self._thread.start()
            self.get_logger().info(
                f"Front camera bridge publishing to {self.camera_topic} on iface={self.camera_interface or 'auto'}"
            )
        except Exception as exc:
            self._thread = None
            self.get_logger().error(f"Failed to initialize VideoClient: {exc}")

    def _camera_loop(self) -> None:
        poll_period = 1.0 / max(self.camera_poll_hz, 0.5)
        if self._client is None:
            return

        while not self._stop.is_set():
            started = time.monotonic()
            try:
                code, data = self._client.GetImageSample()
                if code == 0 and data is not None:
                    msg = self._jpeg_to_image_msg(bytes(data))
                    if msg is not None:
                        self.publisher.publish(msg)
                else:
                    self.get_logger().warning(f"VideoClient returned non-zero code: {code}")
                    time.sleep(0.2)
            except Exception as exc:
                self.get_logger().warning(f"Front camera read failed: {exc}")
                time.sleep(0.5)
                continue

            elapsed = time.monotonic() - started
            remaining = poll_period - elapsed
            if remaining > 0:
                self._stop.wait(remaining)

    def _jpeg_to_image_msg(self, payload: bytes) -> Image | None:
        try:
            image = PillowImage.open(io.BytesIO(payload)).convert("RGB")
        except Exception as exc:
            self.get_logger().warning(f"Front camera decode failed: {exc}")
            return None

        msg = Image()
        msg.header.frame_id = self.camera_frame_id
        msg.width = int(image.width)
        msg.height = int(image.height)
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = int(image.width) * 3
        msg.data = image.tobytes()
        return msg

    def destroy_node(self) -> bool:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = FrontCameraBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
