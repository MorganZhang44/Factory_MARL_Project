from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    topic_prefix = LaunchConfiguration("topic_prefix")
    heartbeat_period = LaunchConfiguration("heartbeat_period")
    core_state_host = LaunchConfiguration("core_state_host")
    core_state_port = LaunchConfiguration("core_state_port")
    core_state_websocket_period = LaunchConfiguration("core_state_websocket_period")
    enable_control_loop = LaunchConfiguration("enable_control_loop")
    control_topic_prefix = LaunchConfiguration("control_topic_prefix")
    control_period = LaunchConfiguration("control_period")
    planning_period = LaunchConfiguration("planning_period")
    path_stale_after = LaunchConfiguration("path_stale_after")
    navdp_timeout = LaunchConfiguration("navdp_timeout")
    locomotion_timeout = LaunchConfiguration("locomotion_timeout")
    navdp_url = LaunchConfiguration("navdp_url")
    locomotion_url = LaunchConfiguration("locomotion_url")
    visualization_host = LaunchConfiguration("visualization_host")
    visualization_port = LaunchConfiguration("visualization_port")
    visualization_core_ws_url = LaunchConfiguration("visualization_core_ws_url")

    return LaunchDescription(
        [
            DeclareLaunchArgument("topic_prefix", default_value="/factory/simulation"),
            DeclareLaunchArgument("heartbeat_period", default_value="1.0"),
            DeclareLaunchArgument("core_state_host", default_value="0.0.0.0"),
            DeclareLaunchArgument("core_state_port", default_value="8765"),
            DeclareLaunchArgument("core_state_websocket_period", default_value="0.1"),
            DeclareLaunchArgument("enable_control_loop", default_value="true"),
            DeclareLaunchArgument("control_topic_prefix", default_value="/factory/control"),
            DeclareLaunchArgument("control_period", default_value="0.02"),
            DeclareLaunchArgument("planning_period", default_value="0.5"),
            DeclareLaunchArgument("path_stale_after", default_value="2.0"),
            DeclareLaunchArgument("navdp_timeout", default_value="10.0"),
            DeclareLaunchArgument("locomotion_timeout", default_value="0.08"),
            DeclareLaunchArgument("navdp_url", default_value="http://127.0.0.1:8889"),
            DeclareLaunchArgument("locomotion_url", default_value="http://127.0.0.1:8890"),
            DeclareLaunchArgument("visualization_host", default_value="0.0.0.0"),
            DeclareLaunchArgument("visualization_port", default_value="8770"),
            DeclareLaunchArgument("visualization_core_ws_url", default_value="ws://127.0.0.1:8765/ws"),
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
                        "enable_control_loop": enable_control_loop,
                        "control_topic_prefix": control_topic_prefix,
                        "control_period": control_period,
                        "planning_period": planning_period,
                        "path_stale_after": path_stale_after,
                        "navdp_timeout": navdp_timeout,
                        "locomotion_timeout": locomotion_timeout,
                        "navdp_url": navdp_url,
                        "locomotion_url": locomotion_url,
                    }
                ],
            ),
            Node(
                package="factory_core",
                executable="core_visualization_node",
                name="factory_core_visualization",
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
