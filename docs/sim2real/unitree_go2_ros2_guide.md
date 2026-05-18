# Unitree Go2 + Ubuntu + ROS2 Setup Guide

> This guide summarizes the practical setup path we used for Go2 integration with ROS2 and the Unitree SDK stack.
> Date aligned with the current repository cleanup: 2026-05-18.

## 1. High-Level Architecture

The Go2 high-level motion stack runs on the robot. The user PC connects over wired Ethernet and communicates through CycloneDDS.

Two paths matter in practice:

| Path | Purpose |
|---|---|
| Wired Ethernet + CycloneDDS | high-level motion control, state, LiDAR, and ROS-visible data |
| SDK (`unitree_sdk2_python`) | front camera and direct high-level control helpers such as `SportClient` and `VideoClient` |

For research and debugging, use the wired DDS path as the default baseline.

## 2. Recommended Host Setup

Recommended baseline:

- Ubuntu 22.04
- ROS2 Humble
- CycloneDDS

Typical packages:

```bash
sudo apt install ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosidl-generator-dds-idl \
    ros-humble-rviz2 \
    ros-humble-plotjuggler-ros \
    ros-humble-rqt-graph \
    libyaml-cpp-dev libeigen3-dev libboost-all-dev
```

## 3. Network Configuration

Typical wired layout:

| Item | Value |
|---|---|
| Robot subnet | `192.168.123.0/24` |
| Example PC address | `192.168.123.99` |
| Go2 onboard address used in our setup | `192.168.123.161` |

Example host-side configuration:

```bash
sudo ip addr flush dev eno1
sudo ip addr add 192.168.123.99/24 dev eno1
sudo ip link set eno1 up
```

Verify with:

```bash
ping -c 4 192.168.123.161
```

## 4. CycloneDDS

CycloneDDS must be selected explicitly, and the host interface name must match the wired interface connected to the robot.

Example:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eno1" priority="default" multicast="default" /></Interfaces></General></Domain></CycloneDDS>'
```

If the interface name is wrong, ROS2 discovery will fail even when the link is physically up.

## 5. ROS2 Topics We Actually Care About

Core state:

- `/sportmodestate`
- `/lowstate`

Robot pose and localization:

- `/utlidar/robot_pose`
- `/utlidar/robot_odom`

LiDAR:

- `/utlidar/cloud`
- `/utlidar/cloud_deskewed`
- `/utlidar/imu`

High-level control path:

- `/api/sport/request`
- `/api/sport/response`

Useful first check:

```bash
ros2 topic list
```

## 6. Camera Path

For this project, the front camera is handled through `unitree_sdk2_python` and `VideoClient`, not as a standard ROS image topic.

The practical path is:

```python
from unitree_sdk2py.go2.video.video_client import VideoClient
```

Then decode the returned JPEG bytes into an image for display or forwarding.

## 7. Minimal Motion Control

The simplest control path uses `SportClient`:

```python
from unitree_sdk2py.go2.sport.sport_client import SportClient
```

Typical sequence:

1. initialize channel factory
2. `StandUp()`
3. `BalanceStand()`
4. stream `Move(vx, vy, vyaw)` commands
5. `StopMove()`
6. `Damp()`

In this repository, the ready-to-run example is:

```bash
python sim2real/scripts/go2_forward_back_test.py --iface eno1
```

## 8. Relationship to `sim2real/`

This repository does not expose the raw Unitree setup directly as the main user-facing entry point. Instead, the cleaned-up workflow is:

```text
Go2 DDS + SDK
  -> sim2real/
  -> dashboard and motion tests
```

So for day-to-day usage, prefer:

```bash
cd sim2real
./scripts/rebuild_env.sh
conda activate sim2real
./scripts/launch_dashboard.sh
```

That path already wraps the repository-specific environment, ROS message build, dashboard launch, and camera integration strategy.
