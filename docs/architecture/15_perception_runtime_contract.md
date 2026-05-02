# Perception Runtime Contract

## Purpose

This document records the current formal runtime contract for the `perception`
module.

It focuses on three things:

* what `perception` is allowed to consume
* how `perception` is synchronized with `simulation`
* what data-path decisions were explicitly rejected

This is a runtime integration note for the current project state.

---

## Position in the System

The current runtime path is:

```text
Simulation -> Core -> Perception -> Core
```

Important:

* `perception` does not talk directly to `simulation`
* `perception` does not talk directly to `navdp`
* `perception` does not talk directly to `locomotion`
* all input reaches `perception` through `core`

This follows the project-wide rule that functional modules only communicate
through the Core Communication Layer.

---

## Environment Ownership

`perception` runs only in its own environment:

```text
perception
```

It must not rely on:

* the `isaaclab51` environment
* the `core` environment
* the `navdp` environment
* the `locomotion` environment

Runtime communication is through the Perception HTTP adapter only.

Current adapter entry point:

```text
perception/perception_service.py
```

Default port:

```text
8891
```

---

## Allowed Inputs

`perception` is a sensor interpretation module.

It is allowed to consume:

* robot pose metadata from `core`
* robot IMU
* robot LiDAR point cloud
* robot front camera RGB / depth / semantic segmentation
* fixed CCTV RGB / semantic segmentation
* simulation timestamps and step indices
* simulation ground truth only for diagnostics / evaluation

It is not allowed to consume low-level policy internals as perception inputs.

---

## Explicitly Rejected Input

The following input path is now explicitly rejected:

```text
simulation locomotion_observation -> core -> perception
```

Reason:

* `locomotion_observation` is a low-level control / policy observation vector
* it belongs to the locomotion chain, not the perception chain
* letting `perception` consume it creates hidden coupling between modules
* it breaks the architectural separation between sensing and control

This means:

* `perception` must not depend on policy observation layout
* `perception` must not read body velocity from locomotion observation
* `perception` must not use locomotion-only features as sensor substitutes

The observation vector is still valid for the locomotion module and dashboard
telemetry, but it is not part of the formal Perception runtime input contract.

---

## Current Perception Sensor Inputs

### 1. IMU

Source:

```text
/factory/simulation/{robot_id}/imu
```

Message type:

```text
sensor_msgs/Imu
```

Current usage in `perception`:

* angular velocity
* linear acceleration
* orientation

Orientation is also used to derive projected gravity in body frame inside the
Perception adapter.

Current rule:

* projected gravity for `perception` must come from IMU orientation or another
  perception-owned sensor derivation
* it must not come from the locomotion observation vector

### 2. LiDAR

Source:

```text
/factory/simulation/{robot_id}/lidar/points
```

Message type:

```text
sensor_msgs/PointCloud2
```

Current usage in `perception`:

* dog self-localization scan-to-map matching
* intruder dynamic cluster detection

### 3. Dog Front Camera

Sources:

```text
/factory/simulation/{robot_id}/camera/image_raw
/factory/simulation/{robot_id}/camera/depth
/factory/simulation/{robot_id}/camera/semantic_segmentation
```

Current usage in `perception`:

* semantic intruder detection
* monocular / depth-assisted target positioning

### 4. CCTV

Sources:

```text
/factory/simulation/cctv/{camera_id}/image_raw
/factory/simulation/cctv/{camera_id}/semantic_segmentation
```

Current usage in `perception`:

* global intruder detection from fixed cameras

### 5. Ground Truth

Ground truth is still mirrored into the Perception adapter for:

* evaluation
* debugging
* error measurement

It must not be used as a direct measurement input to the localization or target
estimation logic.

---

## Time Base Rule

The Perception module must follow the `simulation` timeline, not wall-clock
runtime.

Reason:

* current Isaac runtime may have very low real-time factor
* if perception integrates motion using wall-clock cadence, velocity prediction
  becomes wrong when simulation runs slower than real time

Formal rule:

* `simulation` is the source of step index
* `core` converts simulation step index to simulation time using `simulation_dt`
* `perception` updates use this simulation timestamp as the authoritative time
  axis

Current representation:

```text
sim_time = sim_step * simulation_dt
```

Where:

* `sim_step` comes from the Simulation aggregate state timestamp field
* `simulation_dt` is a Core launch/runtime parameter

---

## Perception Scheduling Rule

`core` may call `perception` asynchronously, but the call frequency is evaluated
on the simulation timeline.

That means:

* `perception_period` is measured in simulated seconds
* not in wall-clock seconds

Example:

* `perception_period = 0.04`

means:

* one Perception update every `0.2` simulated seconds

not:

* one update every `0.2` real seconds

This prevents low RTF from silently changing perception behavior.

---

## Dog Localization Time Rule

Dog self-localization inside `perception` must use dynamic `dt` derived from
the incoming simulation timestamps.

Current behavior:

* first update may use a default bootstrap `dt`
* later updates compute:

```text
dt = current_timestamp - last_timestamp
```

This `dt` is then used in:

* IMU propagation
* velocity integration
* LiDAR correction timing
* yaw-rate and bias-related consistency checks

This replaces the earlier fixed-rate assumption.

---

## Consequences of the Current Contract

### Good

* `perception` is now cleaner as a module boundary
* low RTF no longer changes the meaning of the Perception update cadence
* perception no longer depends on locomotion policy observation schema
* sensor ownership is easier to reason about

### Tradeoff

Because the locomotion observation shortcut was removed:

* dog velocity estimation in `perception` now depends more directly on
  IMU quality, timestamp consistency, and LiDAR correction quality

This is expected and architecturally correct.

---

## Files That Currently Define This Contract

Primary runtime files:

* [core/ros2/factory_core/factory_core/control_node.py](/home/yyz/projects/Factory_MARL_Project/core/ros2/factory_core/factory_core/control_node.py)
* [perception/perception_service.py](/home/yyz/projects/Factory_MARL_Project/perception/perception_service.py)
* [perception/perception/pipeline.py](/home/yyz/projects/Factory_MARL_Project/perception/perception/pipeline.py)
* [perception/perception/dog_localizer.py](/home/yyz/projects/Factory_MARL_Project/perception/perception/dog_localizer.py)
* [simulation/standalone/validate_slam_scene.py](/home/yyz/projects/Factory_MARL_Project/simulation/standalone/validate_slam_scene.py)

Related architecture references:

* [1_modules.md](/home/yyz/projects/Factory_MARL_Project/docs/architecture/1_modules.md)
* [10_ros2_sim_core_topics.md](/home/yyz/projects/Factory_MARL_Project/docs/architecture/10_ros2_sim_core_topics.md)
* [12_runtime_environments.md](/home/yyz/projects/Factory_MARL_Project/docs/architecture/12_runtime_environments.md)

---

## Summary

The current project decision is:

* `perception` consumes only perception-appropriate simulation sensor data
* `perception` does not consume locomotion policy observation vectors
* `perception` follows the simulation time axis
* `core` remains the only routing hub between `simulation` and `perception`

This is the current runtime contract and should be treated as the baseline
unless a later document explicitly replaces it.
