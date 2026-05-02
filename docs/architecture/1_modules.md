# Modules

## Overview

This document defines all core modules in the system, including their roles, inputs, outputs, and interaction patterns.

All modules:

* communicate **only through the Core Communication Layer**
* do NOT directly call each other
* run only in their owned runtime environment
* operate within a **synchronous execution loop (Version 1)**

---

## Module List

* Simulation
* Perception / Target Estimation
* Decision Making
* MARL Decision Module
* NavDP Path Planning
* Locomotion
* Core Communication Layer
* Logger / Replay / Evaluation

---

## 1. Simulation

### Role

* Provide the Isaac Sim / Isaac Lab simulation environment
* Maintain ground-truth state of:

  * agents
  * intruder
  * map / obstacles
* Control episode lifecycle (reset, step, termination)

### Input (via Communication Layer)

* `locomotion/motion_command`
* control signals (reset, pause)

### Output (via Communication Layer)

* `simulation/state`

### Notes

* Acts as **single source of truth**
* Runs only in the `isaaclab51` environment
* Uses USD scenes and Isaac Sim / Isaac Lab APIs directly
* In Version 1, may expose intruder ground truth for development convenience
* Responsible for applying actions at each timestep

---

## 2. Perception / Target Estimation

### Role

* Convert raw simulation sensor data into **dog pose estimates** and **intruder pose estimates**
* Handle visibility, uncertainty, and sensor fusion

### Input

* `simulation/state` (robot poses, intruder pose, sensor data)
* Robot IMU, LiDAR, camera, CCTV semantic segmentation

### Output

* `perception/target_estimate` (intruder position, velocity, confidence)
* Dog pose estimates (position, velocity, orientation, localization status)

### Notes

* Now an independent module with its own runtime environment (`perception`)
* Runs as an HTTP adapter on port `8891`
* Current implementation includes:

  * **Dog self-localization**: IMU propagation + LiDAR scan-to-map correction
  * **Intruder detection**: CCTV semantic segmentation with monocular ground-plane projection, dog camera semantic/depth detections, LiDAR dynamic cluster detection
  * **Sensor fusion**: robust camera/LiDAR fusion with Kalman tracking over `[x, y, vx, vy]`
* Ground truth is available for evaluation/logging only, not used as measurement input
* Perception does **not** consume locomotion policy observation vectors
* Core calls Perception asynchronously through Core only
* `perception_period` is interpreted on the simulation timeline, not wall-clock time
* Default `perception_period = 0.04` means one Perception update every `0.04` simulated seconds

---

## 3. Decision Making

### Role

* Generate coordinated strategy for agents
* Core module for MARL

### Input

* `simulation/state`
* `perception/target_estimate`

### Output

* `decision/subgoal`

### Notes

* Produces **one subgoal per agent per timestep**
* Initial version may use rule-based logic
* Later replaced by MARL policy

---

## 4. MARL Decision Module

### Role

* Provide a standalone multi-agent decision policy runtime
* Convert mirrored multi-agent state into one subgoal per agent
* Act as the concrete runtime candidate for the Decision layer

### Input

* `agent_1` world position / velocity
* `agent_2` world position / velocity
* `intruder_1` world position / velocity

### Output

* one world-frame `subgoal` per agent

### Notes

* Runs only in the `marl` environment
* Current service adapter runs on port `8892`
* Current action semantics are:

  * policy outputs **world-frame relative offsets**
  * adapter converts them into **world-frame subgoals**
* Current integration stage:

  * Core mirrors MARL inputs and outputs for visualization
  * MARL does **not** yet replace the active control path by default

---

## 5. NavDP Path Planning

### Role

* Convert subgoals into executable paths
* Generate waypoint sequences
* Handle obstacle avoidance

### Input

* `decision/subgoal`
* `simulation/state` (for map / obstacle info)

### Output

* `planning/path`

### Notes

* Path is a sequence of waypoints
* Initial version may use:

  * straight-line
  * simple grid-based planning

---

## 6. Locomotion

### Role

* Execute movement commands
* Convert paths into motion control

### Input

* `planning/path`
* `simulation/state`

### Output

* `locomotion/motion_command`

### Notes

* Motion command is **velocity-based (vx, vy)**
* Simplified control in Version 1

---

## 7. Core Communication Layer

### Role

* Central middleware for all module interaction

### Responsibilities

* message routing (topic-based)
* publish / subscribe mechanism
* message validation (basic)
* enforcing interface schema

### Input

* messages published by modules

### Output

* routed messages to subscribers

---

### Communication Model (Version 1)

* **Publish–Subscribe**
* Topic-based routing
* Broadcast to all subscribers of a topic

---

### Execution Model

* Runs in the `core` environment
* Acts as the central message dispatcher and runtime coordinator
* Receives Simulation data through ROS2
* Exposes Core-owned state to Visualization through REST / WebSocket

---

### Non-Responsibilities (Important)

The communication layer does NOT:

* perform any domain logic
* modify payload content
* make decisions
* maintain simulation state

---

## 8. Logger / Replay / Evaluation

### Role

* Record system behavior
* Support replay and evaluation

### Input

* subscribes to **all topics**

### Output

* logs
* replay data
* evaluation metrics

### Notes

* Minimal logging sufficient in Version 1
* Extended evaluation added later

---

## Summary

The system consists of modular components connected via a **Core Communication Layer**.

Each module:

* consumes well-defined topics
* produces structured messages
* runs in its own runtime environment
* operates independently

This design ensures:

* modularity
* scalability
* compatibility with industrial architectures
