from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    topic_prefix = LaunchConfiguration("topic_prefix")
    publish_rate = LaunchConfiguration("publish_rate")
    heartbeat_period = LaunchConfiguration("heartbeat_period")
    enable_visualization = LaunchConfiguration("enable_visualization")
    core_state_host = LaunchConfiguration("core_state_host")
    core_state_port = LaunchConfiguration("core_state_port")
    core_state_websocket_period = LaunchConfiguration("core_state_websocket_period")
    visualization_host = LaunchConfiguration("visualization_host")
    visualization_port = LaunchConfiguration("visualization_port")
    visualization_core_ws_url = LaunchConfiguration("visualization_core_ws_url")

    return LaunchDescription(
        [
            DeclareLaunchArgument("topic_prefix", default_value="/factory/simulation"),
            DeclareLaunchArgument("publish_rate", default_value="10.0"),
            DeclareLaunchArgument("heartbeat_period", default_value="1.0"),
            DeclareLaunchArgument("enable_visualization", default_value="false"),
            DeclareLaunchArgument("core_state_host", default_value="0.0.0.0"),
            DeclareLaunchArgument("core_state_port", default_value="8765"),
            DeclareLaunchArgument("core_state_websocket_period", default_value="0.1"),
            DeclareLaunchArgument("visualization_host", default_value="0.0.0.0"),
            DeclareLaunchArgument("visualization_port", default_value="8770"),
            DeclareLaunchArgument("visualization_core_ws_url", default_value="ws://127.0.0.1:8765/ws"),
            Node(
                package="factory_sim_bridge",
                executable="mock_sim_publisher",
                name="factory_mock_sim_publisher",
                output="screen",
                parameters=[
                    {
                        "topic_prefix": topic_prefix,
                        "publish_rate": publish_rate,
                    }
                ],
            ),
            Node(
                package="factory_core",
                executable="core_control_node",
                name="factory_core_control",
                output="screen",
                parameters=[
                    {
                        "topic_prefix": topic_prefix,
                        "heartbeat_period": heartbeat_period,
                        "state_host": core_state_host,
                        "state_port": core_state_port,
                        "state_websocket_period": core_state_websocket_period,
                    }
                ],
            ),
            Node(
                package="factory_core",
                executable="core_visualization_node",
                name="factory_core_visualization",
                output="screen",
                condition=IfCondition(enable_visualization),
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
