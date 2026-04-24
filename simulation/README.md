# Simulation Module

The Simulation module owns Isaac Sim / Isaac Lab scene setup and Simulation-side
ROS2 publishing.

It must run only in the `isaaclab51` environment.

Primary local entry point:

```bash
./scripts/launch_simulation.sh
```

This launches the Isaac Sim / Isaac Lab scene validator in
`simulation/standalone/validate_slam_scene.py`.

By default the standalone Isaac Sim entry point publishes the current
Simulation-Core ROS2 contract under `/factory/simulation`.

Current ROS2-facing package:

```text
simulation/ros2/factory_sim_bridge
```

Responsibilities in Version 1:

* publish robot poses
* publish intruder pose
* publish robot camera images
* publish robot LiDAR scans
* publish robot LiDAR point clouds
* publish a lightweight aggregate debug state

Current status:

* robot and intruder poses are read from Isaac Sim articulation state
* camera frames are read from Isaac Lab camera buffers
* LiDAR `/scan` and `/points` are read from Isaac Sim RTX LiDAR annotators
* the dashboard currently visualizes the LaserScan projection of the real RTX
  LiDAR data

The current package includes `mock_sim_publisher` so the Core layer can be
tested before Isaac Sim publishes real sensor data.

The mock publisher is not the primary Simulation runtime.
