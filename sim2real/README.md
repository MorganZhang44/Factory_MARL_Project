# Sim2Real

`sim2real/` is now a standalone real-robot integration line for Unitree Go2.
Its goal is to bring real robot state, sensors, and camera streams into the
project’s existing `core + dashboard` style, then grow outward toward more
advanced control and module integration.

The directory is intentionally organized so that you can move the entire
`sim2real/` folder as one self-contained subproject:

- startup scripts live in `sim2real/scripts/`
- ROS2 packages live in `sim2real/ros2/`
- a local Unitree ROS2 copy lives in `sim2real/unitree_ros2/`
- a local Unitree Python SDK copy lives in `sim2real/unitree_sdk2_python/`

The runtime no longer depends on top-level project modules such as
`simulation/`, `marl/`, `navdp/`, `locomotion/`, or `perception/`.

## Structure

- `ros2/factory_core_sim2real`
  - sim2real-specific Core derivative
  - subscribes to real robot topics, aggregates state, and exposes `/api/state`
    plus a websocket feed
- `ros2/factory_bringup_sim2real`
  - sim2real-specific launch package
- `unitree_ros2`
  - local copy of the official ROS2 message / example repository
- `unitree_sdk2_python`
  - local copy of the official Python SDK
- `scripts/go2_forward_back_test.py`
  - minimal real-robot motion validation script

## Connected Data

### First batch

- `/sportmodestate`
- `/utlidar/robot_pose`
- `/utlidar/robot_odom`

### Second batch

- `/utlidar/imu`
- `/utlidar/cloud`

### Camera

The camera path does not decode `/frontvideostream` directly. It uses the
officially recommended `VideoClient` path instead:

- `camera_worker.py` uses
  `unitree_sdk2py.go2.video.video_client.VideoClient`
- it pulls JPEG frames periodically
- it writes the latest frame to:
  - `/tmp/factory_sim2real/front_camera.jpg`
- `core_control_node` reads that cached image and feeds it to the dashboard

This split is intentional:

- `VideoClient` is stable when run on its own
- mixing `VideoClient` and `rclpy` in the same ROS2 node can conflict around
  `ChannelFactoryInitialize`
- a dedicated worker is easier to debug and more reliable

## Dashboard Capabilities

The current sim2real dashboard shows:

- robot pose
- `sportmodestate`
- IMU state
- LiDAR top-down point-cloud view
  - height-colored
- front camera image

Default dashboard address:

- `http://127.0.0.1:8770/`

Core state API:

- `http://127.0.0.1:8765/`

## Environment Setup

Recommended setup:

```bash
cd sim2real
./scripts/rebuild_env.sh
conda activate sim2real
```

`sim2real/environment.yml` defines the conda base environment.
`sim2real/requirements.txt` adds pip-side dependencies.

If the environment already exists:

```bash
cd sim2real
pip install -r requirements.txt
```

## Launching the Dashboard

```bash
conda activate sim2real
cd sim2real
./scripts/launch_dashboard.sh
```

This script will:

1. activate the `sim2real` conda environment
2. configure CycloneDDS
3. auto-build the local `unitree_ros2` message workspace if
   `unitree_ros2/cyclonedds_ws/install/setup.bash` is missing
4. source that `setup.bash`
5. build the local `ros2` workspace
6. launch `factory_bringup_sim2real`

Default network interface:

- `eno1`

Override example:

```bash
SIM2REAL_NET_IFACE=enp6s0 ./scripts/launch_dashboard.sh
```

## Building the Unitree ROS2 Message Workspace

If the local `unitree_ros2` copy needs rebuilding:

```bash
conda activate sim2real
cd sim2real
./scripts/build_unitree_ros2.sh
```

The repository intentionally does not keep generated
`unitree_ros2/cyclonedds_ws/build/install/log` or `ros2/workspace/build/install/log`
artifacts. `launch_dashboard.sh` will rebuild what it needs.

## Minimal Motion Test

The current minimal validation script is:

- `scripts/go2_forward_back_test.py`

It performs:

1. `StandUp()`
2. `BalanceStand()`
3. forward motion at `0.1 m/s` for `1 s`
4. backward motion at `0.1 m/s` for `1 s`
5. `StopMove()`
6. `Damp()`

Run:

```bash
conda activate sim2real
cd sim2real
python scripts/go2_forward_back_test.py --iface eno1
```

Skip stand-up if the robot is already standing:

```bash
python scripts/go2_forward_back_test.py --iface eno1 --skip-standup
```

## Key Files

- `ros2/factory_core_sim2real/factory_core_sim2real/control_node.py`
- `ros2/factory_core_sim2real/factory_core_sim2real/state_mirror.py`
- `ros2/factory_core_sim2real/factory_core_sim2real/visualization_node.py`
- `ros2/factory_core_sim2real/factory_core_sim2real/camera_worker.py`
- `ros2/factory_bringup_sim2real/launch/core_dashboard.launch.py`
- `scripts/launch_dashboard.sh`

## Current Facts

- default point-cloud source:
  - `/utlidar/cloud`
- LiDAR frame:
  - `utlidar_lidar`
- front camera currently retrieved through the SDK as:
  - `1920 x 1080`
  - `jpeg`
- the dashboard should currently show:
  - `pose=True`
  - `status=True`
  - `camera=True`
  - `imu=True`
  - `lidar_points=True`

## Natural Next Steps

- add `/wirelesscontroller` to the UI for manual takeover visibility
- add snapshot / recording support for camera and point clouds
- wrap `SportClient` control more cleanly inside the sim2real runtime
- decide whether higher-level modules should be connected to the real-robot line
