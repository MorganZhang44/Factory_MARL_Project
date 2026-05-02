# ROS2 Simulation-Core Topics

This is the Version 1 ROS2 contract between Simulation and the Core
Communication Layer.

The goal is to keep the first integration simple and compatible with standard
ROS2 tooling. Custom message definitions can be introduced later when the
schemas become stable.

---

## Topic Prefix

Default prefix:

```text
/factory/simulation
```

All Simulation-to-Core topic names below are relative to this prefix.

---

## Entity Poses

### Robot Pose

```text
/{robot_id}/pose
```

Message type:

```text
geometry_msgs/PoseStamped
```

Current robot IDs:

```text
agent_1
agent_2
```

Full topic examples:

```text
/factory/simulation/agent_1/pose
/factory/simulation/agent_2/pose
```

---

### Intruder Pose

```text
/{intruder_id}/pose
```

Message type:

```text
geometry_msgs/PoseStamped
```

Current intruder ID:

```text
intruder_1
```

Full topic example:

```text
/factory/simulation/intruder_1/pose
```

---

## Robot Sensors

### Front Camera

```text
/{robot_id}/camera/image_raw
```

Message type:

```text
sensor_msgs/Image
```

Full topic examples:

```text
/factory/simulation/agent_1/camera/image_raw
/factory/simulation/agent_2/camera/image_raw
```

### Front Camera Depth

```text
/{robot_id}/camera/depth
```

Message type:

```text
sensor_msgs/Image
```

Encoding:

```text
32FC1
```

Full topic examples:

```text
/factory/simulation/agent_1/camera/depth
/factory/simulation/agent_2/camera/depth
```

### Front Camera Semantic Segmentation

```text
/{robot_id}/camera/semantic_segmentation
```

Message type:

```text
sensor_msgs/Image
```

Encoding:

```text
32SC1
```

Full topic examples:

```text
/factory/simulation/agent_1/camera/semantic_segmentation
/factory/simulation/agent_2/camera/semantic_segmentation
```

---

### Front LiDAR

```text
/{robot_id}/lidar/scan
```

Message type:

```text
sensor_msgs/LaserScan
```

Full topic examples:

```text
/factory/simulation/agent_1/lidar/scan
/factory/simulation/agent_2/lidar/scan
```

### Front LiDAR Point Cloud

```text
/{robot_id}/lidar/points
```

Message type:

```text
sensor_msgs/PointCloud2
```

Full topic examples:

```text
/factory/simulation/agent_1/lidar/points
/factory/simulation/agent_2/lidar/points
```

The formal runtime LiDAR is the same Isaac Lab RayCaster profile used by
Perception: 16 channels, `-45` to `45` degree vertical FOV, full 360 degree
horizontal FOV, 1 degree horizontal resolution, `(0, 0, 0.35)` dog-base offset,
50 m max distance, and `/World/LocalizationStaticMesh` as the ray-cast target.
The `/lidar/points` topic is built from RayCaster hit points. The `/lidar/scan`
topic is a 2D projection of the same RayCaster point set so existing Core /
Dashboard debug visualization can keep using LaserScan.

### Dog IMU

```text
/{robot_id}/imu
```

Message type:

```text
sensor_msgs/Imu
```

Full topic examples:

```text
/factory/simulation/agent_1/imu
/factory/simulation/agent_2/imu
```

### CCTV Cameras

Perception-aligned fixed CCTV camera IDs:

```text
cam_nw
cam_ne
cam_e_upper
cam_e_lower
cam_se
cam_sw
```

RGB topic:

```text
/cctv/{camera_id}/image_raw
```

Semantic segmentation topic:

```text
/cctv/{camera_id}/semantic_segmentation
```

Message type:

```text
sensor_msgs/Image
```

Full topic examples:

```text
/factory/simulation/cctv/cam_nw/image_raw
/factory/simulation/cctv/cam_nw/semantic_segmentation
```

### Locomotion Observation

```text
/{robot_id}/locomotion/observation
```

Message type:

```text
std_msgs/String
```

Full topic examples:

```text
/factory/simulation/agent_1/locomotion/observation
/factory/simulation/agent_2/locomotion/observation
```

Payload:

```json
{
  "robot_id": "agent_1",
  "timestamp": 0,
  "schema": "go2_flat_velocity_policy_obs_v1",
  "observation": [48 values]
}
```

The 48D observation follows the Unitree Go2 flat velocity policy training order:
base linear velocity, base angular velocity, projected gravity, commanded base
velocity, relative joint position, relative joint velocity, and last action.

---

## Aggregate Debug State

```text
/state
```

Message type:

```text
std_msgs/String
```

Payload:

```json
{
  "timestamp": 0.0,
  "frame_id": "world",
  "robots": {
    "agent_1": {"position": [x, y, z]},
    "agent_2": {"position": [x, y, z]}
  },
  "intruders": {
    "intruder_1": {"position": [x, y, z]}
  }
}
```

This topic is for debugging and rapid integration only. High-frequency sensor
data should stay on typed ROS2 topics.

Core subscribes to all formal robot sensor topics:

* `/camera/image_raw`
* `/camera/depth`
* `/camera/semantic_segmentation`
* `/imu`
* `/lidar/scan`
* `/lidar/points`
* `/cctv/<camera_id>/image_raw`
* `/cctv/<camera_id>/semantic_segmentation`

The Core state mirror exposes a lightweight dashboard snapshot for RGB, depth
statistics, semantic-label summaries, IMU samples, 2D LiDAR scan points,
downsampled 3D LiDAR points, and CCTV RGB feeds. The control payload sent from
Core to upper/lower modules may include the latest RGB image, depth image,
semantic image, IMU, LaserScan, and PointCloud2 payloads. Modules that do not
need a sensor should ignore the corresponding key rather than requiring a
different topic contract.

---

## Core-to-Simulation Control

Default control prefix:

```text
/factory/control
```

### Motion Command

```text
/locomotion/motion_command
```

Message type:

```text
std_msgs/String
```

Full topic:

```text
/factory/control/locomotion/motion_command
```

Payload envelope:

```json
{
  "message_id": "core-control-...",
  "timestamp": 0.0,
  "topic": "locomotion/motion_command",
  "source_module": "core",
  "payload": {
    "commands": {
      "agent_1": {
        "velocity": [vx, vy],
        "action": [12 values],
        "action_scale": 0.25,
        "subgoal": [x, y],
        "path": [[x1, y1], [x2, y2]]
      }
    }
  }
}
```

Rules:

* velocity is world-frame, in m/s
* action is the 12D low-level Unitree Go2 joint action
* action is applied as `default_joint_position + action_scale * action`
* Simulation ignores stale commands
* Simulation applies commands only through this ROS2 topic
* NavDP and Locomotion never talk to Simulation directly

---

## Core Web Dashboard

The Core Communication Layer provides a read-only state API for system-level
observability. The visualization frontend must read from this Core API, not
directly from Simulation ROS2 topics.

Current entry points:

```bash
./scripts/launch_simulation.sh
./scripts/launch_core_dashboard.sh
```

Simulation must run in the `isaaclab51` environment and should use Isaac Sim /
Isaac Lab directly from the `simulation/` module. Core and the visualization
frontend run together in the `core` environment.

Temporary mock ROS2 publishers may be used for interface testing, but they are
not the primary Simulation runtime.

Default Core state API:

```text
http://localhost:8765
```

Default visualization frontend:

```text
http://localhost:8770
```

The frontend is read-only and visualizes:

* robot world positions
* intruder world position
* latest robot camera frames
* latest robot LiDAR scans
* latest robot 3D LiDAR point-cloud summary
* latest robot depth image summary
* topic freshness / stale status

### Data Access

Real-time updates use WebSocket:

```text
ws://localhost:8765/ws
```

Low-frequency queries use REST:

```text
GET /api/state
GET /health
```

### Non-Blocking Rule

The visualization path must not block the main control path.

Required behavior:

* dashboard reads a state summary or mirrored cache
* dashboard does not subscribe to Simulation ROS2 topics
* dashboard does not sit inside the control loop
* WebSocket/REST failures must not stop Core control
* browser rendering speed must not affect ROS2 subscriptions or control routing

Core owns system status, debugging, and observability. Visualization is a client
of that Core-owned state, not another ROS2 data consumer.
