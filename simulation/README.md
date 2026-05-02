# Simulation Module

The Simulation module owns Isaac Sim / Isaac Lab scene setup and Simulation-side
ROS2 publishing.

It must run only in the `isaaclab51` environment.

Primary local entry point:

```bash
./scripts/launch_simulation.sh
```

The launcher now supports two runtimes:

```bash
./scripts/launch_simulation.sh --runtime legacy
./scripts/launch_simulation.sh --runtime rewrite
```

Or with an environment variable:

```bash
SIMULATION_RUNTIME=rewrite ./scripts/launch_simulation.sh
```

The launcher uses `conda run --no-capture-output` with `PYTHONUNBUFFERED=1`, so
Isaac Sim logs and Python tracebacks should appear directly in the terminal.

Runtime mapping:

* `legacy`: launches `simulation/standalone/validate_slam_scene.py`
* `rewrite`: launches `simulation/standalone/run_environment_rewrite.py`

By default the standalone Isaac Sim entry point publishes the current
Simulation-Core ROS2 contract under `/factory/simulation`.

Because both runtimes publish the same current-project topic contract, the rest
of the stack does not need to change when switching between them.

Current ROS2-facing package:

```text
simulation/ros2/factory_sim_bridge
```

Responsibilities in Version 1:

* publish robot poses
* publish intruder pose
* publish robot camera images
* publish robot camera depth images
* publish robot camera semantic segmentation images
* publish robot IMU samples
* publish robot LiDAR scans
* publish robot LiDAR point clouds
* publish fixed CCTV camera images and semantic segmentation images
* publish a lightweight aggregate debug state

Current status:

* robot and intruder poses are read from Isaac Sim articulation state
* camera frames are read from Isaac Lab camera buffers
* depth frames are read from Isaac Lab camera buffers as `32FC1`
* semantic segmentation frames are read from Isaac Lab camera buffers as `32SC1`
* LiDAR `/scan` and `/points` are generated from the same Isaac Lab RayCaster
  profile used by `perception/environment`
* the formal LiDAR profile is 16 channels, 360 degree horizontal coverage,
  `-45` to `45` degree vertical coverage, 1 degree horizontal resolution,
  `(0, 0, 0.35)` mount offset, and 50 m max range
* fixed CCTV cameras follow the perception camera layout:
  `cam_nw`, `cam_ne`, `cam_e_upper`, `cam_e_lower`, `cam_se`, `cam_sw`
* the dashboard currently visualizes the LaserScan projection of the RayCaster
  point set

The current package includes `mock_sim_publisher` so the Core layer can be
tested before Isaac Sim publishes real sensor data.

The mock publisher is not the primary Simulation runtime.
