from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_id = LaunchConfiguration("robot_id")
    topic_prefix = LaunchConfiguration("topic_prefix")
    sport_mode_topic = LaunchConfiguration("sport_mode_topic")
    robot_pose_topic = LaunchConfiguration("robot_pose_topic")
    robot_odom_topic = LaunchConfiguration("robot_odom_topic")
    imu_topic = LaunchConfiguration("imu_topic")
    lidar_points_topic = LaunchConfiguration("lidar_points_topic")
    camera_topic = LaunchConfiguration("camera_topic")
    camera_interface = LaunchConfiguration("camera_interface")
    camera_poll_hz = LaunchConfiguration("camera_poll_hz")
    camera_frame_id = LaunchConfiguration("camera_frame_id")
    camera_cache_path = LaunchConfiguration("camera_cache_path")
    stale_after = LaunchConfiguration("stale_after")
    core_state_host = LaunchConfiguration("core_state_host")
    core_state_port = LaunchConfiguration("core_state_port")
    core_state_websocket_period = LaunchConfiguration("core_state_websocket_period")
    visualization_host = LaunchConfiguration("visualization_host")
    visualization_port = LaunchConfiguration("visualization_port")
    visualization_core_ws_url = LaunchConfiguration("visualization_core_ws_url")

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_id", default_value="agent_1"),
            DeclareLaunchArgument("topic_prefix", default_value="/factory/sim2real"),
            DeclareLaunchArgument("sport_mode_topic", default_value="/sportmodestate"),
            DeclareLaunchArgument("robot_pose_topic", default_value="/utlidar/robot_pose"),
            DeclareLaunchArgument("robot_odom_topic", default_value="/utlidar/robot_odom"),
            DeclareLaunchArgument("imu_topic", default_value="/utlidar/imu"),
            DeclareLaunchArgument("lidar_points_topic", default_value="/utlidar/cloud"),
            DeclareLaunchArgument("camera_topic", default_value="/factory/sim2real/agent_1/camera/image_raw"),
            DeclareLaunchArgument("camera_interface", default_value="eno1"),
            DeclareLaunchArgument("camera_poll_hz", default_value="8.0"),
            DeclareLaunchArgument("camera_frame_id", default_value="front_camera"),
            DeclareLaunchArgument("camera_cache_path", default_value="/tmp/factory_sim2real/front_camera.jpg"),
            DeclareLaunchArgument("stale_after", default_value="1.0"),
            DeclareLaunchArgument("core_state_host", default_value="0.0.0.0"),
            DeclareLaunchArgument("core_state_port", default_value="8765"),
            DeclareLaunchArgument("core_state_websocket_period", default_value="0.1"),
            DeclareLaunchArgument("visualization_host", default_value="0.0.0.0"),
            DeclareLaunchArgument("visualization_port", default_value="8770"),
            DeclareLaunchArgument("visualization_core_ws_url", default_value="ws://127.0.0.1:8765/ws"),
            Node(
                package="factory_core_sim2real",
                executable="core_control_node",
                name="factory_core_sim2real_control",
                output="screen",
                parameters=[
                    {
                        "robot_id": robot_id,
                        "topic_prefix": topic_prefix,
                        "sport_mode_topic": sport_mode_topic,
                        "robot_pose_topic": robot_pose_topic,
                        "robot_odom_topic": robot_odom_topic,
                        "camera_topic": camera_topic,
                        "camera_cache_path": camera_cache_path,
                        "camera_frame_id": camera_frame_id,
                        "imu_topic": imu_topic,
                        "lidar_points_topic": lidar_points_topic,
                        "stale_after": stale_after,
                        "state_host": core_state_host,
                        "state_port": core_state_port,
                        "state_websocket_period": core_state_websocket_period,
                    }
                ],
            ),
            Node(
                package="factory_core_sim2real",
                executable="camera_worker",
                name="factory_core_sim2real_camera_worker",
                output="screen",
                arguments=[
                    "--camera-interface",
                    camera_interface,
                    "--camera-poll-hz",
                    camera_poll_hz,
                    "--camera-cache-path",
                    camera_cache_path,
                ],
            ),
            Node(
                package="factory_core_sim2real",
                executable="core_visualization_node",
                name="factory_core_sim2real_visualization",
                output="screen",
                parameters=[
                    {
                        "web_host": visualization_host,
                        "web_port": visualization_port,
                        "core_ws_url": visualization_core_ws_url,
                    }
                ],
            ),
        ]
    )
