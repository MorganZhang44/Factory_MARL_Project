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
    perception_timeout = LaunchConfiguration("perception_timeout")
    marl_timeout = LaunchConfiguration("marl_timeout")
    navdp_url = LaunchConfiguration("navdp_url")
    locomotion_url = LaunchConfiguration("locomotion_url")
    perception_url = LaunchConfiguration("perception_url")
    marl_url = LaunchConfiguration("marl_url")
    perception_period = LaunchConfiguration("perception_period")
    perception_record_dir = LaunchConfiguration("perception_record_dir")
    marl_period = LaunchConfiguration("marl_period")
    use_marl_output = LaunchConfiguration("use_marl_output")
    simulation_dt = LaunchConfiguration("simulation_dt")
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
            DeclareLaunchArgument("perception_timeout", default_value="0.4"),
            DeclareLaunchArgument("marl_timeout", default_value="0.15"),
            DeclareLaunchArgument("navdp_url", default_value="http://127.0.0.1:8889"),
            DeclareLaunchArgument("locomotion_url", default_value="http://127.0.0.1:8890"),
            DeclareLaunchArgument("perception_url", default_value="http://127.0.0.1:8891"),
            DeclareLaunchArgument("marl_url", default_value="http://127.0.0.1:8892"),
            DeclareLaunchArgument("perception_period", default_value="0.04"),
            DeclareLaunchArgument("perception_record_dir", default_value=""),
            DeclareLaunchArgument("marl_period", default_value="0.1"),
            DeclareLaunchArgument("use_marl_output", default_value="true"),
            DeclareLaunchArgument("simulation_dt", default_value="0.005"),
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
                        "perception_timeout": perception_timeout,
                        "marl_timeout": marl_timeout,
                        "navdp_url": navdp_url,
                        "locomotion_url": locomotion_url,
                        "perception_url": perception_url,
                        "marl_url": marl_url,
                        "perception_period": perception_period,
                        "perception_record_dir": perception_record_dir,
                        "marl_period": marl_period,
                        "use_marl_output": use_marl_output,
                        "simulation_dt": simulation_dt,
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
