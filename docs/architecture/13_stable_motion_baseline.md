# Stable Motion Baseline

## Purpose

This document records the manually restored version that is currently able to
follow the planned route correctly.

It is a runtime baseline, not a future-looking design document.

The goal is simple:

* keep one clear reference for the last known good control chain
* make later regressions easier to locate
* separate "stable baseline behavior" from later experiments on visualization,
  telemetry, and performance

Current reference date:

* baseline confirmed on `2026-04-23`
* behavior: robots can follow the NavDP path without the later jittering issue

---

## Scope

This baseline describes the currently restored working chain:

```text
Simulation -> Core -> NavDP -> Core -> Locomotion -> Core -> Simulation
```

And the attached observability path:

```text
Simulation -> Core State Mirror -> Visualization
```

Important:

* Visualization is present, but it is not treated as part of the control loop.
* This document describes what the code does now, not what the final system
  should eventually become.

---

## Environment Ownership

This baseline still follows the per-module environment rule.

* `simulation` runs in `isaaclab51`
* `core` and visualization run in `core`
* `perception` runs in `perception`
* ROS2 bringup runs from the ROS2 workspace launched by the core side
* `navdp` runs in its own module environment
* `locomotion` runs in its own module environment

Cross-module communication in this baseline is explicit:

* `simulation <-> core`: ROS2 topics
* `core <-> perception`: HTTP
* `core <-> navdp`: HTTP
* `core <-> locomotion`: HTTP
* `visualization <- core`: WebSocket / state API

---

## Files That Define This Baseline

Primary runtime behavior is defined by:

* [simulation/standalone/validate_slam_scene.py](/home/yyz/projects/Factory_MARL_Project/simulation/standalone/validate_slam_scene.py)
* [core/ros2/factory_core/factory_core/control_node.py](/home/yyz/projects/Factory_MARL_Project/core/ros2/factory_core/factory_core/control_node.py)
* [navdp/navdp_service.py](/home/yyz/projects/Factory_MARL_Project/navdp/navdp_service.py)
* [locomotion/locomotion_service.py](/home/yyz/projects/Factory_MARL_Project/locomotion/locomotion_service.py)
* [ros2/factory_bringup/launch/core_dashboard.launch.py](/home/yyz/projects/Factory_MARL_Project/ros2/factory_bringup/launch/core_dashboard.launch.py)
* [scripts/launch_simulation.sh](/home/yyz/projects/Factory_MARL_Project/scripts/launch_simulation.sh)
* [scripts/launch_core_dashboard.sh](/home/yyz/projects/Factory_MARL_Project/scripts/launch_core_dashboard.sh)

---

## Stable Runtime Parameters

### Simulation

From `validate_slam_scene.py`:

* simulation timestep: `dt = 0.005`
* ROS2 publish decimation: `publish_every = 4`
* default command timeout: `max_command_age = 1.0`
* default world-frame command scale: `command_scale = 1.0`
* front camera enabled
* RTX LiDAR enabled

This gives the effective observation / control publication rhythm:

* physics: `200 Hz`
* ROS2 publish rhythm: `50 Hz`

### Core

From `control_node.py`:

* control loop period: `0.02 s`
* planning period: `0.5 s`
* path stale timeout: `2.0 s`
* NavDP timeout: `10.0 s`
* Locomotion timeout: `0.08 s`

This means:

* Core control loop runs at `50 Hz`
* NavDP replans at most at `10 Hz`
* Core reuses a cached path between planning calls

### Locomotion

From `locomotion_service.py`:

* max speed: `0.8`
* lookahead: `0.35`
* stop distance: `0.15`
* low-level action scale: `0.25`

---

## Control Chain Semantics

### 1. Simulation publishes robot state and observations

Simulation publishes:

* robot pose
* intruder pose
* RGB image
* depth image
* LiDAR scan
* LiDAR point cloud
* locomotion observation
* aggregate simulation state

The locomotion observation is a 48D vector built in
`_make_locomotion_observation()`:

* base linear velocity in body frame: 3
* base angular velocity in body frame: 3
* projected gravity: 3
* command slot placeholder: 3
* joint position relative to default pose: 12
* joint velocity relative to default velocity: 12
* previous low-level action: 12

Schema label:

* `go2_flat_velocity_policy_obs_v1`

### 2. Core receives simulation data and keeps the control loop simple

In this stable version, Core:

* stores `locomotion_observation` as raw JSON
* does not decode it into extra telemetry fields
* does not add extra low-level policy feature switches
* does not reshape the observation for visualization-side inference

Per robot, Core:

1. reads robot pose and intruder pose
2. creates a chase subgoal using the intruder XY position
3. computes `local_goal` in the robot body frame
4. asynchronously asks NavDP for a path
5. derives a body-frame velocity command from the path
6. sends both the path and the raw locomotion observation to Locomotion
7. publishes one ROS2 `motion_command` message back to Simulation

### 3. NavDP returns a path, not low-level control

NavDP remains path-planning-only in this baseline.

Core sends:

* `robot_state`
* `subgoal`
* `local_goal`
* `robot_yaw`
* camera/depth payload when present
* simulation state

NavDP returns:

* world-frame `waypoints`
* optionally `local_waypoints`
* planner identity

If the real planner is available, local path points are transformed back into
world coordinates using `robot_yaw`.

### 4. Core derives the command that goes into the low-level policy

This stable version uses `_body_command_from_navdp_path()`.

Priority is:

* if `local_waypoints` exist, use NavDP's local trajectory directly
* otherwise, fall back to world path -> body-frame conversion

The output is a body-frame command:

* `[vx, vy, wz]`

Current implementation keeps:

* `wz = 0.0`

### 5. Locomotion uses the low-level policy

This baseline is not a pure velocity-only fallback.

When `locomotion_observation` is present, Locomotion:

1. loads the 48D observation
2. overwrites `obs[9:12]` with the incoming `body_velocity_command`
3. runs `Go2ActorPolicy`
4. returns a 12D low-level action

Returned payload includes:

* `velocity`
* `action`
* `action_scale`
* `target`
* `controller = go2_low_level_policy_v1`

The key semantic point is:

* the low-level policy is active in this stable baseline
* command injection happens through observation indices `9:12`

### 6. Simulation gives low-level action priority

In `_apply_motion_commands()`:

* low-level joint action takes priority over root velocity fallback
* root velocity is only used when no low-level action is available

Low-level application is:

```text
joint_target = default_joint_pos + action_scale * action
```

And then:

* `dog.set_joint_position_target(joint_target)`

For robots without low-level action in the current tick:

* Simulation falls back to direct root translation / velocity writing

This priority order is part of the stable behavior and should be preserved when
debugging regressions.

---

## ROS2 Topic Contract Used In This Baseline

Simulation publishes under:

* `/factory/simulation/state`
* `/factory/simulation/<robot_id>/pose`
* `/factory/simulation/<robot_id>/camera/image_raw`
* `/factory/simulation/<robot_id>/camera/depth`
* `/factory/simulation/<robot_id>/lidar/scan`
* `/factory/simulation/<robot_id>/lidar/points`
* `/factory/simulation/<robot_id>/locomotion/observation`
* `/factory/simulation/<intruder_id>/pose`

Core publishes back to Simulation under:

* `/factory/control/locomotion/motion_command`

The motion command payload may contain:

* `velocity`
* `action`
* `action_scale`
* `subgoal`
* `body_velocity_command`
* `path`

---

## Why This Version Is a Useful Baseline

This version is important because it is the last known version where:

* NavDP path following is working
* low-level policy is active
* robots can move along the planned route without the later regression

That makes it the right comparison point for:

* jitter regressions
* action / observation interface regressions
* visualization-side telemetry additions
* performance-tuning edits that accidentally change runtime behavior

---

## Things This Baseline Does Not Include

Compared with later experimental versions, this baseline does not include the
following control-path changes:

* no `enable_locomotion_policy` feature flag in Core
* no extra decoding of locomotion observation into dashboard-only telemetry
* no `joint_target_rel` / `joint_target_abs` additions in the control payload
* no defensive clipping of low-level action inside Core / Simulation
* no extra policy-output post-processing fields such as `path_target`
* no special decimation-boundary latching logic added after the fact

It also does not include the later performance-tuning CLI flags that were added
for benchmarking-oriented runs.

That separation matters:

* this file is meant to preserve the stable motion reference
* later tuning and observability changes should be compared against it

---

## Practical Rule For Future Changes

When changing any of the following, compare behavior against this baseline:

* `locomotion_observation` schema or ordering
* command injection into observation slots
* action scaling
* low-level action application timing
* Core control loop timing
* visualization-related telemetry extraction inside Core

If route following breaks again, first check whether the current code still
matches the semantics described in this document before tuning the policy or the
scene.
