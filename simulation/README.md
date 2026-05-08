# Simulation Module

The Simulation module owns Isaac Sim / Isaac Lab scene setup and Simulation-side
ROS2 publishing.

It must run only in the `isaaclab51` environment.

## Primary entry point

```bash
./scripts/launch_simulation.sh --runtime legacy --keep-open
```

The launcher supports three runtimes:

```bash
./scripts/launch_simulation.sh --runtime legacy
./scripts/launch_simulation.sh --runtime rewrite
./scripts/launch_simulation.sh --runtime rebuild
```

Or with an environment variable:

```bash
SIMULATION_RUNTIME=legacy ./scripts/launch_simulation.sh
```

The launcher uses `conda run --no-capture-output` with `PYTHONUNBUFFERED=1`, so
Isaac Sim logs and Python tracebacks should appear directly in the terminal.

Runtime mapping:

* `legacy`: launches `simulation/standalone/validate_slam_scene.py`
* `rewrite`: launches `simulation/standalone/run_environment_rewrite.py`
* `rebuild`: launches `simulation/standalone/run_environment_rebuild.py`

Current recommendation:

* use `legacy` for end-to-end stack tests
* treat `rewrite` and `rebuild` as parallel lines

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

## Legacy runtime notes

`legacy` is the current baseline runtime for full-stack testing.

Useful flags:

```bash
./scripts/launch_simulation.sh --runtime legacy --keep-open
./scripts/launch_simulation.sh --runtime legacy --keep-open --move-intruder
```

Behavior:

* by default the `intruder` is held fixed in place
* with `--move-intruder`, it follows the fixed route used by the current
  project setup

Current default actor spawn positions:

* `agent_1 = (-2.0, -2.0, 0.42)`
* `agent_2 = (-2.0, 1.6, 0.42)`
* `intruder = (2.0, -0.5, 1.34)`

## Docker

The repository also includes a headless Isaac Sim container path:

```text
simulation/Dockerfile.headless
```

In `compose.yaml`, the `simulation` service currently runs with:

* `simulation/Dockerfile.headless`
* `--headless`

So the Docker simulation path is currently for headless stack testing, not GUI
visual debugging.

The current package includes `mock_sim_publisher` so the Core layer can be
tested before Isaac Sim publishes real sensor data.

The mock publisher is not the primary Simulation runtime.
