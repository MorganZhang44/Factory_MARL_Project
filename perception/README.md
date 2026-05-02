# Multi-Agent Surveillance: Environment + Perception Architecture

This repository simulates a multi-agent surveillance scene in NVIDIA Isaac Lab and estimates global poses for:

- two Unitree Go2 robot dogs: `go2_dog_1`, `go2_dog_2`
- one moving intruder/suspect: `intruder` / `suspect`

The codebase is now split into two explicit packages:

- `environment/`: simulation-facing package. It owns Isaac scene access, actor motion, raw sensor collection, and optional ground-truth diagnostics.
- `perception/`: algorithm package. It consumes `environment` sensor outputs and produces global pose estimates for the dogs and intruder.

The intended dependency direction is:

```text
Isaac Lab scene
   |
   v
environment package
   |
   |  EnvironmentSensorFrame
   |  - dog IMU outputs
   |  - dog LiDAR outputs
   |  - CCTV and dog camera outputs
   |  - optional motion hints
   |  - optional ground-truth diagnostics
   v
perception package
   |
   |  PerceptionOutput
   |  - dog pose estimates
   |  - intruder pose estimate
   |  - detection metadata
   |  - optional evaluation errors
   v
logs / plots / HTTP pose server / downstream programs
```

The perception package should not read Isaac scene objects directly. Its input is the sensor packet emitted by the environment layer.

---

## Runtime

This project runs the current perception adapter as an HTTP service behind:

```text
perception/perception_service.py
```

Local launch:

```bash
./scripts/launch_perception.sh
```

Docker launch:

```bash
docker compose up --build perception
```

The Docker image for this module is:

```text
perception/Dockerfile
```

The Docker path preserves the same ownership rule as the local path:

* module: `perception`
* environment: `perception`
* public interface: HTTP `/health` and `/estimate`

---

## Directory Layout

```text
.
├── main.py
│   Compatibility wrapper for `environment.main`.
│
├── environment/
│   ├── __init__.py
│   ├── main.py
│   │   Isaac Lab entry point. Wires simulation, perception, logging,
│   │   visualization, and optional HTTP pose serving.
│   ├── types.py
│   │   Dataclass contracts for raw sensor output:
│   │   EnvironmentSensorFrame, CameraSensorOutput, LidarSensorOutput,
│   │   ImuSensorOutput, DogMotionHint, GroundTruthState.
│   ├── isaac_bridge.py
│       Isaac Lab adapter. Converts InteractiveScene sensors into
│       EnvironmentSensorFrame objects.
│   ├── config/
│   │   Isaac Lab scene, camera, LiDAR, and IMU configuration.
│   ├── scene/
│   │   Actor path generation, scripted motion, and ground-truth utilities.
│   ├── static_scene_geometry.py
│   │   Shared map geometry, walkable polygon, camera positions, obstacles.
│   └── localization_mesh.py
│       Static mesh helper for LiDAR localization.
│
├── perception/
│   ├── __init__.py
│   ├── types.py
│   │   Dataclass contracts for pose output:
│   │   PerceptionOutput, DogPoseEstimate, IntruderPoseEstimate.
│   ├── pipeline.py
│   │   High-level perception API:
│   │   EnvironmentSensorFrame -> PerceptionOutput.
│   ├── dog_localizer.py
│   │   Dog self-localization: IMU propagation + LiDAR scan-to-map correction.
│   ├── scan_matching.py
│   │   Static map generation and trimmed ICP scan matching.
│   ├── camera_detector.py
│   │   Semantic segmentation/depth/monocular CCTV detector for intruder.
│   ├── lidar_detector.py
│   │   Dog LiDAR dynamic-cluster detector for intruder.
│   ├── fusion.py
│       Robust camera/LiDAR fusion and Kalman tracking for intruder.
│   ├── pose_server.py
│   │   Lightweight HTTP server for live pose output.
│   ├── trajectory_plots.py
│   ├── dog_pointcloud_video.py
│   ├── visualization.py
│   └── transforms.py
│       Perception-side math and output helpers.
│
└── test_*.py
    Unit tests for dog localization, surveillance fusion, path planning, and
    pose server behavior.
```

---

## Environment Package

The environment package is responsible for raw simulation output. It does not decide where the intruder or dogs are; it only publishes sensor data.

### Main Types

`EnvironmentSensorFrame` is the main output of the environment layer.

```python
from environment.types import EnvironmentSensorFrame
```

Fields:

```text
step: int
timestamp: float
dog_imus: dict[str, ImuSensorOutput]
dog_lidars: dict[str, LidarSensorOutput]
cameras: dict[str, CameraSensorOutput]
dog_motion_hints: dict[str, DogMotionHint]
ground_truth: dict[str, GroundTruthState]
```

`dog_imus` contains one entry per dog:

```text
go2_dog_1:
  ang_vel_b              body-frame angular velocity
  lin_acc_b              body-frame linear acceleration
  projected_gravity_b    gravity direction projected into body frame
  timestamp              simulation time
  odom_vel_w             optional commanded/odometry velocity in world frame
  odom_ang_vel_w         optional commanded/odometry angular velocity in world frame
```

`dog_lidars` contains one entry per dog LiDAR:

```text
dog1_lidar:
  hit_points             ray hit points from Isaac Lab RayCaster
  pos_w                  sensor position in world frame
  quat_w                 sensor orientation in world frame
```

`cameras` contains CCTV and dog camera data:

```text
cam_sw / cam_ne / dog1_cam / ...
  semantic_segmentation
  depth                  None for fixed CCTV, available for dog cameras
  rgb
  info                   semantic segmentation metadata
  intrinsic_matrix
  pos_w
  quat_w
```

`ground_truth` is optional and simulation-only. It is useful for metrics and API diagnostics, but perception algorithms should not use it as a measurement.

### Isaac Bridge

`IsaacEnvironmentBridge` converts an Isaac `InteractiveScene` into environment frames:

```python
from environment.isaac_bridge import IsaacEnvironmentBridge

bridge = IsaacEnvironmentBridge(scene)
frame = bridge.collect_frame(
    step=step,
    timestamp=sim_time,
    motion_hints=scene_manager.get_dog_motion_hints(),
    include_lidar=True,
    include_cameras=True,
    include_ground_truth=True,
)
```

The bridge also exposes lower-level collectors:

```python
bridge.collect_dog_imus(timestamp, motion_hints)
bridge.collect_lidars()
bridge.collect_cameras()
bridge.collect_camera_rgb()
bridge.collect_ground_truth()
```

### Environment Responsibilities

The environment layer may:

- spawn and move actors
- read Isaac camera/LiDAR/IMU tensors
- expose command odometry hints used by dog localization
- expose raw sensor frames
- expose ground truth for logging and evaluation

The environment layer should not:

- run semantic detection
- run ICP
- fuse sensors
- estimate intruder or dog pose
- apply localization filters

---

## Perception Package

The perception package consumes environment outputs and produces global pose estimates.

### High-Level API

```python
from perception.pipeline import PerceptionPipeline

pipeline = PerceptionPipeline(
    camera_intrinsics=camera_intrinsics,
    camera_conventions=camera_conventions,
    dog_initial_poses={
        "go2_dog_1": (dog1_initial_pos, dog1_initial_quat),
        "go2_dog_2": (dog2_initial_pos, dog2_initial_quat),
    },
    dt=1.0 / 60.0,
)

perception_output = pipeline.update(frame)
```

Input:

```text
EnvironmentSensorFrame
```

Output:

```text
PerceptionOutput
  step
  timestamp
  dogs: dict[str, DogPoseEstimate]
  intruder: IntruderPoseEstimate | None
```

### Dog Pose Output

Each dog output wraps the existing `DogLocalizer` dictionary:

```text
DogPoseEstimate
  name
  step
  timestamp
  state:
    pos
    vel
    quat
    euler
    speed
    localized
    lidar_corrected
    only_imu_prediction
    scan_match_score
    scan_inliers
  ground_truth
  xy_error_m
  yaw_error_deg
```

Dog localization uses:

- IMU propagation every simulation step
- commanded dog motion hint as an odometry prior
- dog LiDAR scan-to-map correction when a new LiDAR frame is available
- static map generated from scene geometry

It does not use dog ground-truth pose as an estimate input.

### Intruder Pose Output

The intruder output wraps `FusionResult`:

```text
IntruderPoseEstimate
  step
  timestamp
  fusion_result:
    detected
    position_world
    velocity_world
    confidence
    error_meters
    num_camera_detections
    num_lidar_detections
  camera_detections
  lidar_detections
  ground_truth
```

Intruder localization uses:

- fixed CCTV semantic segmentation with monocular ground-plane projection
- dog camera semantic/depth detections for diagnostics
- dog LiDAR dynamic cluster detections
- robust filtering of impossible CCTV projections
- down-weighting of LiDAR when it disagrees with reliable CCTV consensus
- Kalman tracking over `[x, y, vx, vy]`

### Perception Responsibilities

The perception layer may:

- detect intruder pixels/points from sensor tensors
- run dog IMU/LiDAR self-localization
- fuse camera and LiDAR detections
- output pose estimates, velocity estimates, confidence, and error diagnostics

The perception layer should not:

- step the simulator
- directly read Isaac scene entities
- move actors
- create cameras or sensors

---

## Reproduction Dependencies

The two packages `environment/` and `perception/` are the clean API boundary,
and they now contain the runnable project code. The root directory only keeps
launcher/documentation/dependency files.

For a full replay of this repository, keep these project pieces together:

- `main.py`
- `environment/`
- `perception/`
- `requirements.txt`
- the external Isaac Sim / Isaac Lab installation and assets

Install the normal Python dependencies inside the Isaac-compatible environment:

```bash
/home/ming/anaconda3/envs/env_isaaclab/bin/python -m pip install -r requirements.txt
```

Isaac Sim and Isaac Lab must be installed separately. They are imported by
`environment/main.py`, `environment/config/`, and `environment/isaac_bridge.py`,
but they are not PyPI packages managed by this requirements file.

If another program only wants to feed already-collected sensor packets into the
perception stack, the minimum reusable subset is smaller: `perception/`,
`environment/types.py`, and `environment/static_scene_geometry.py` for scan
matching and map constraints.

---

## Runtime Data Flow in `main.py`

`environment/main.py` is the standalone executable that wires everything
together. The root `main.py` is only a compatibility wrapper, so existing
commands such as `python main.py ...` still work:

```text
1. Launch Isaac Lab AppLauncher
2. Build SurveillanceSceneCfg and InteractiveScene
3. Create SurveillanceSceneManager
4. Collect raw environment sensor outputs
5. Run perception
6. Write logs and plots
7. Optionally publish live pose API
```

For backwards compatibility, some legacy collector helpers still exist in
`environment/main.py`. The package boundary is available through:

```python
environment.isaac_bridge.IsaacEnvironmentBridge
perception.pipeline.PerceptionPipeline
```

New code should prefer those package APIs instead of adding more direct scene reads inside perception.

---

## Live Pose Server

The live pose server is optional. It runs inside `main.py` while the simulation process is alive.

Start it with:

```bash
cd /home/ming/Documents/multiagent

env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
/home/ming/anaconda3/envs/env_isaaclab/bin/python main.py \
  --headless --no_viz --num_steps 999999 \
  --pose_server --pose_host 127.0.0.1 --pose_port 8765
```

Then open:

```text
http://127.0.0.1:8765/
```

The root page is a small live dashboard that polls `/poses` every 250 ms.

If another computer must access the server:

```bash
--pose_host 0.0.0.0
```

Then open:

```text
http://<machine-ip>:8765/
```

### HTTP Endpoints

```text
GET /                 browser dashboard
GET /health           health check
GET /poses            all current pose state
GET /poses/latest     alias for /poses
GET /poses/dogs       both dogs
GET /poses/go2_dog_1  dog 1 only
GET /poses/go2_dog_2  dog 2 only
GET /poses/intruder   intruder only
```

### Pose JSON Schema

Top-level `/poses` response:

```json
{
  "schema_version": 1,
  "updated_wall_time": 1777044342.79,
  "step": 336,
  "sim_time": 5.6,
  "dogs": {
    "go2_dog_1": {},
    "go2_dog_2": {}
  },
  "intruder": {}
}
```

Dog response:

```json
{
  "name": "go2_dog_1",
  "step": 335,
  "time": 5.58,
  "estimate": {
    "pos": [-4.16, -3.36, 0.42],
    "vel": [0.70, 0.00, 0.0],
    "quat": [0.999, 0.0, 0.0, -0.026],
    "euler": {
      "rad": [0.0, 0.0, -0.052],
      "deg": [0.0, 0.0, -3.0]
    },
    "speed": 0.70,
    "localized": true,
    "lidar_corrected": true,
    "only_imu_prediction": false,
    "scan_match_score": 0.93,
    "scan_inliers": 150
  },
  "ground_truth": {
    "pos": [-4.15, -3.35, 0.42],
    "vel": [0.70, 0.0, 0.0],
    "quat": [1.0, 0.0, 0.0, 0.0],
    "euler": {
      "rad": [0.0, 0.0, 0.0],
      "deg": [0.0, 0.0, 0.0]
    },
    "speed": 0.70
  },
  "errors": {
    "xy_m": 0.02,
    "yaw_deg": 1.2
  }
}
```

Intruder response:

```json
{
  "name": "intruder",
  "step": 335,
  "time": 5.58,
  "estimate": {
    "pos": [4.42, 0.79, 0.9],
    "vel": [0.81, 0.33, 0.0],
    "quat": [0.98, 0.0, 0.0, 0.19],
    "euler": {
      "rad": [0.0, 0.0, 0.38],
      "deg": [0.0, 0.0, 21.95]
    },
    "speed": 0.87,
    "heading_from_velocity": true,
    "detected": true,
    "confidence": 0.31,
    "num_camera_detections": 5,
    "num_lidar_detections": 2
  },
  "ground_truth": {
    "pos": [4.40, 0.78, 1.34],
    "vel": [0.80, 0.30, 0.0],
    "angular_vel": [0.0, 0.0, 0.0],
    "quat": [1.0, 0.0, 0.0, 0.0],
    "euler": {
      "rad": [0.0, 0.0, 0.0],
      "deg": [0.0, 0.0, 0.0]
    },
    "speed": 0.85
  },
  "errors": {
    "xy_m": 0.12
  }
}
```

The HTTP server only exists while `main.py` is running. If the simulation exits, the webpage and API stop too.

---

## Logs and Artifacts

Default output directory:

```text
output/
```

Important files:

```text
output/localization_log.csv
  Intruder estimate vs ground truth per perception update.

output/dog_localization_log.csv
  Dog estimate vs ground truth per simulation step.

output/trajectory_comparison.png
  Intruder estimated trajectory vs ground truth.

output/dog_trajectory_comparison.png
  Dog estimated trajectories vs ground truth.

output/surveillance_video.avi
  CCTV/dog camera tiled video.

output/dog_pointcloud_video.avi
  Dog LiDAR point-cloud visualization.

output/intruder_detection_debug.csv
  Optional debug CSV enabled by DEBUG_INTRUDER_STEPS=start:end.
```

Enable intruder raw detection diagnostics:

```bash
DEBUG_INTRUDER_STEPS=2860:3030 \
env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
/home/ming/anaconda3/envs/env_isaaclab/bin/python main.py \
  --headless --no_viz --num_steps 3065
```

---

## Running

### Headless With Pose Server

```bash
cd /home/ming/Documents/multiagent

env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
/home/ming/anaconda3/envs/env_isaaclab/bin/python main.py \
  --headless --no_viz --num_steps 3900 \
  --pose_server --pose_port 8765
```

### GUI Run

```bash
env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
/home/ming/anaconda3/envs/env_isaaclab/bin/python main.py \
  --num_steps 1200 \
  --pose_server --pose_port 8765
```

### Long-Running API Run

Use a very large step count if you want the webpage/API to stay alive:

```bash
env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
/home/ming/anaconda3/envs/env_isaaclab/bin/python main.py \
  --headless --no_viz --num_steps 999999 \
  --pose_server --pose_host 0.0.0.0 --pose_port 8765
```

### Console Pose Snapshots

```bash
--pose_print_interval 120
```

This prints live dog and intruder pose snapshots every 120 simulation steps.

---

## Testing

Run all focused tests:

```bash
env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
/home/ming/anaconda3/envs/env_isaaclab/bin/python -m unittest -v \
  test_pose_server.py \
  test_dog_localization.py \
  test_scene_paths.py \
  test_surveillance_tracking.py
```

Current tested areas:

- pose HTTP server
- dog IMU/LiDAR localization
- scan matching
- walkable path planning
- intruder sensor fusion robustness
- LiDAR/camera disagreement handling

---

## Current Performance Snapshot

Recent full `3900`-step validation:

```text
Intruder:
  mean XY error:   0.202 m
  median XY error: 0.149 m
  P95 XY error:    0.582 m
  max XY error:    1.255 m

Dogs:
  go2_dog_1 mean XY error: about 0.17 m
  go2_dog_2 mean XY error: about 0.18 m
```

These values are simulation metrics and depend on route, camera visibility, and occlusion.

---

## Design Rules Going Forward

Keep this boundary clean:

```text
environment -> perception -> outputs
```

Good:

```python
frame = bridge.collect_frame(...)
output = pipeline.update(frame)
```

Avoid:

```python
# Bad: perception reaching into Isaac scene directly
dog = scene["go2_dog_1"]
```

Use ground truth only for:

- logs
- metrics
- plots
- HTTP diagnostics

Do not use ground truth as a perception measurement input.
