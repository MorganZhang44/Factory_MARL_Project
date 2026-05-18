# 06 · Sim2Real and Real-Robot Integration

**Primary directory:** `sim2real/`  
**Primary launch script:** `sim2real/scripts/launch_dashboard.sh`  
**Primary motion test:** `sim2real/scripts/go2_forward_back_test.py`

## What `sim2real/` Is

`sim2real/` is now a standalone real-robot subproject. It is not an extension of the research demo tree and it is not meant to be mixed into MARL training scripts.

Its job is to:

- connect to a real Go2 over the project network setup
- read robot state and sensors
- render a dashboard in the same style as the main runtime stack
- provide a minimal control path for safe validation

## Current Data Path

```text
Go2 topics / SDK
  -> sim2real core
  -> state mirror
  -> web dashboard
```

## Inputs Currently Wired In

State:

- `/sportmodestate`
- `/utlidar/robot_pose`
- `/utlidar/robot_odom`

Sensors:

- `/utlidar/imu`
- `/utlidar/cloud`
- front camera through `VideoClient`

## Why Camera Uses `VideoClient`

The front camera is not treated as a clean standard ROS image source in this project. Instead, a dedicated worker pulls JPEG frames with the Unitree Python SDK and feeds them into the dashboard pipeline. This keeps the robot-side camera path stable without forcing `VideoClient` and the ROS node stack into the same process.

## What the Dashboard Shows

The current real-robot dashboard displays:

- robot pose
- sport mode state
- IMU data
- front camera image
- LiDAR top-down point cloud
- height-colored point rendering

## Minimal Startup

```bash
cd sim2real
./scripts/rebuild_env.sh
conda activate sim2real
./scripts/launch_dashboard.sh
```

## Minimal Motion Test

```bash
cd sim2real
conda activate sim2real
python scripts/go2_forward_back_test.py --iface eno1
```

## Relationship to the Research Tree

- `marl/research/` is for training, offline evaluation, and controlled demos.
- `sim2real/` is for Go2 integration, real sensing, dashboarding, and minimal runtime control.

Keeping those concerns separate makes the repository easier to maintain and much easier to move between development machines and robot-facing machines.
