# Key Architecture Decisions

## Overview

This document summarizes the current high-level architecture decisions for the
Factory MARL project.

The goal is to keep the system modular while leaving implementation-level
interface details flexible for now.

The system is organized around:

* an independent Simulation layer
* a central Core Communication Layer
* independent Perception, Decision, Planning, and Locomotion modules
* asynchronous logging, replay, and evaluation support
* Docker-based environment management

---

# 1. Core Engineering Principles

## Principle 1: Simulation Is Infrastructure

Simulation is an independent infrastructure layer.

It provides the environment, robot execution, sensors, ground truth, and episode
lifecycle. It is not considered part of the learning or decision-making
algorithm.

---

## Principle 2: Core Communication Layer Is the System Hub

The Core Communication Layer is the only central hub in the system.

It is responsible for:

* communication
* routing
* orchestration
* system-level state management
* health monitoring
* visualization support
* experiment and configuration management

It should not implement the core perception, decision, planning, or locomotion
algorithms.

---

## Principle 3: Functional Modules Are Independent

The following modules should remain independent:

* Perception / Target Estimation
* Decision Making
* NavDP Path Planning
* Locomotion
* Logger / Replay / Evaluation

Each module should be replaceable without requiring major changes to the rest of
the system.

---

## Principle 4: Simulation Uses ROS2

Communication between Simulation and the Core Communication Layer should use
ROS2.

Recommended usage:

* high-frequency state and sensor data: ROS2 topics
* low-frequency control requests: ROS2 services

Examples of low-frequency control requests:

* reset
* pause
* resume
* scene or experiment configuration

---

## Principle 5: Internal Modules Use gRPC

Communication between the Core Communication Layer and the main functional
modules should primarily use gRPC.

This applies to:

* Perception / Target Estimation
* Decision Making
* NavDP Path Planning
* Locomotion

Version 1 may use either frame-triggered calls or streaming, depending on what is
simpler to integrate.

---

## Principle 6: Locomotion Only Talks to the Core Layer

Locomotion should only communicate with the Core Communication Layer.

It should not directly call:

* Simulation
* Decision Making
* NavDP Path Planning
* Perception / Target Estimation

This keeps the execution layer replaceable and prevents hidden dependencies.

---

## Principle 7: NavDP Handles Path Planning Only

NavDP Path Planning is responsible for path planning.

It should not be responsible for:

* low-level action generation
* joint control
* torque control
* locomotion execution

Those responsibilities belong to the Locomotion module.

---

## Principle 8: Shared Interfaces Should Be Versioned

All cross-module interfaces should eventually be managed in a versioned
`shared_interfaces` package or directory.

This should include:

* message definitions
* service definitions
* gRPC proto files
* shared data types
* interface version metadata

Exact schemas can be finalized later.

---

## Principle 9: Docker Manages Environment, Config Manages Experiments

Docker should be used to keep module environments reproducible.

Compose should be used to keep startup behavior consistent.

Configuration files should be used to keep experiments reproducible.

In short:

* Docker manages runtime environments
* Compose manages multi-module startup
* Config files manage experiments

---

## Principle 10: Modules Run Only in Their Own Environments

Each module must be launched only from its owned runtime environment.

Current ownership:

* Simulation: `isaaclab51`
* Core Communication Layer: `core`
* Visualization: `core`
* ROS2 tooling / bringup: `ros2`
* Perception: `perception`
* future Decision Making: `decision`
* NavDP Path Planning: `navdp`
* Locomotion: `locomotion`
* future Logger / Replay / Evaluation: `logger`

No module should rely on another module's Conda environment, Python packages, or
local source tree at runtime. Cross-module interaction must happen through ROS2,
gRPC, REST/WebSocket, or another explicit public interface.

Visualization currently shares the `core` environment because it is a Core-owned
observability surface rather than an independent algorithm module.

---

# 2. Module Responsibilities

## 2.1 Simulation

### Role

Simulation is the independent simulation infrastructure layer.

It is responsible for:

* running Isaac Sim / Isaac Lab in the `isaaclab51` environment
* loading and managing USD scenes
* executing robot dynamics
* generating sensor data
* providing ground truth
* receiving control commands
* driving robot motion
* managing episode lifecycle

Simulation does not belong to any algorithm module.

The primary Simulation runtime is Isaac Sim / Isaac Lab inside the
`simulation/` module. Temporary mock publishers may be used only for integration
testing before the Isaac Sim bridge is ready.

---

### Receives

Simulation may receive:

* robot motion commands
* joint or body control commands
* reset / pause / resume requests
* scene configuration
* experiment configuration

---

### Sends

Simulation may send:

* camera data
* LiDAR / laser data
* IMU data
* robot state
* world state
* target ground truth
* episode state

---

### Communication

Simulation communicates with the Core Communication Layer through ROS2.

Recommended split:

* ROS2 topics for high-frequency state and sensor data
* ROS2 services for reset / pause / resume and other control requests

---

## 2.2 Core Communication Layer

### Role

The Core Communication Layer is the central system hub.

It is responsible for:

* receiving data from Simulation through ROS2
* standardizing incoming data
* routing messages to internal modules
* maintaining common metadata such as timestamp, frame, robot_id, and episode_id
* managing system state
* checking module health
* organizing the control chain
* collecting module outputs
* sending final execution commands back to Simulation
* supporting visualization and debugging
* supporting experiment and configuration management

It does not implement the core algorithms.

---

### Internal Functional Areas

The Core Communication Layer may be understood as containing several internal
functional areas:

* Communication / Adapter
* Router / Dispatcher
* Supervisor / Runtime Manager
* Visualization / Observability
* Experiment / Config Management

These are internal responsibilities, not necessarily separate top-level modules.

---

### Receives

The Core Communication Layer receives:

* ROS2 observations and state from Simulation
* target estimates from Perception / Target Estimation
* middle-level decisions from Decision Making
* planned paths from NavDP Path Planning
* execution outputs or status from Locomotion
* heartbeat, error, and status information from all modules
* experiment control requests from the user or system

---

### Sends

The Core Communication Layer sends:

* standardized observations to Perception / Target Estimation
* state summaries and target state to Decision Making
* subgoals or task instructions to NavDP Path Planning
* planned paths or target trajectories to Locomotion
* final control commands to Simulation
* system status, debug information, alerts, and runtime summaries to visualization tools
* events and runtime logs to Logger / Replay / Evaluation

---

### Communication

Recommended communication patterns:

* with Simulation: ROS2
* with Perception / Target Estimation: gRPC
* with Decision Making: gRPC
* with NavDP Path Planning: gRPC
* with Locomotion: gRPC
* with Logger / Replay / Evaluation: asynchronous write to file, database, or object storage
* with visualization frontend: WebSocket from the Core-owned state API
* with experiment management tools: internal management API or REST/admin API

Visualization is implemented as a Web Dashboard. The dashboard reads mirrored
state from the Core Communication Layer's state API and must not subscribe to
Simulation ROS2 topics directly. REST may be used for low-frequency queries or
filtering, while WebSocket is used for live updates. Visualization remains
outside the main control chain, and failures must not affect control-loop
execution.

---

## 2.3 Perception / Target Estimation

### Role

Perception / Target Estimation is the first algorithmic layer after standardized
system observations.

It is responsible for:

* fusing multi-source observations (CCTV cameras, dog cameras, dog LiDAR, IMU)
* estimating dog poses via IMU propagation + LiDAR scan-to-map correction
* estimating intruder position in the unified world frame
* estimating intruder velocity
* producing short-term prediction when the intruder is temporarily lost
* outputting confidence and visibility state
* supporting future generalization across environments, sensor counts, and sensor layouts

It estimates both dog poses and intruder state. It does not make decisions.

---

### Current Implementation

Perception is now an independent module with its own runtime environment
(`perception` conda env) and HTTP adapter (`perception_service.py` on port 8891).

The perception pipeline consists of:

* **Dog self-localization** (`DogLocalizer`): IMU propagation + LiDAR scan-to-map ICP correction
* **Intruder camera detection** (`CameraDetector`): semantic segmentation with monocular ground-plane projection from CCTV and dog cameras
* **Intruder LiDAR detection** (`LidarDetector`): dynamic cluster detection from dog-mounted LiDAR
* **Sensor fusion** (`SensorFusion`): robust camera/LiDAR fusion with Kalman tracking over `[x, y, vx, vy]`

The module exposes a `PerceptionPipeline` API:

```python
from perception.pipeline import PerceptionPipeline
pipeline = PerceptionPipeline(...)
output = pipeline.update(frame)  # EnvironmentSensorFrame -> PerceptionOutput
```

---

### Receives

Perception / Target Estimation receives via Core HTTP payloads:

* robot poses (position, orientation)
* dog camera RGB, depth, semantic segmentation images
* dog LiDAR point clouds
* dog IMU data (angular velocity, linear acceleration)
* dog locomotion observation (projected gravity, base velocity)
* CCTV camera RGB and semantic segmentation images
* intruder pose (for ground-truth evaluation only)

These inputs are standardized by the Core Communication Layer before being
sent to the module.

---

### Sends

Perception / Target Estimation sends:

* dog pose estimates (position, velocity, orientation, localization status, scan match quality)
* intruder position estimate
* intruder velocity estimate
* detection confidence
* number of camera and LiDAR detections
* input summary (which sensors were present)
* timestamp

---

### Communication

Perception / Target Estimation communicates with the Core Communication Layer
through HTTP (Version 1).

Core calls the Perception adapter asynchronously at a configurable
`perception_period` (default 0.04s, i.e. 25 Hz).

Future versions may migrate to gRPC streaming.

---

## 2.4 Decision Making

### Role

Decision Making is a neutral middle-level decision layer.

At this stage, it should not be tightly bound to MARL. MARL can be one future
implementation of this module.

It is responsible for:

* receiving target state and robot state
* making team-level task decisions
* producing middle-level targets for each robot
* outputting role assignment, priority, and task mode

Only external responsibilities are fixed for now. Internal implementation can be
rule-based, optimization-based, MARL-based, or hybrid.

---

### Receives

Decision Making may receive:

* current robot states
* target position estimate and prediction
* map summary / reachable-area information
* team state summary
* historical context summary, if needed

---

### Sends

Decision Making sends:

* subgoal / semantic waypoint for each robot
* role assignment
* priority
* current task mode
* other middle-level decision state

---

### Communication

Decision Making communicates with the Core Communication Layer through gRPC.

---

## 2.5 NavDP Path Planning

### Role

NavDP Path Planning is the planning layer.

It is responsible for:

* receiving middle-level goals
* using robot state and environment representation for path planning
* outputting paths, waypoint sequences, or local planning results

It is not responsible for action generation or low-level control.

---

### Receives

NavDP Path Planning may receive:

* current robot state
* local environment representation
* subgoal / waypoint from Decision Making
* planning constraints
* local map or reachable-area representation

---

### Sends

NavDP Path Planning sends:

* path
* waypoint sequence
* local planned trajectory
* planning status
* planning confidence or validity status, if needed

---

### Communication

NavDP Path Planning communicates with the Core Communication Layer through gRPC.

---

## 2.6 Locomotion

### Role

Locomotion replaces the earlier idea of a simple Joint Command Adapter.

It is responsible for:

* receiving path, waypoint, or trajectory outputs from planning
* converting planning results into executable robot motion-control outputs
* handling motion execution logic
* reporting execution state and feedback

Locomotion is the motion execution layer. It does not perform global path
planning.

---

### Receives

Locomotion may receive:

* path or waypoint sequence from NavDP Path Planning
* current robot state
* execution mode
* execution constraints or control context from the Core Communication Layer

---

### Sends

Locomotion may send:

* robot motion-control commands
* joint-level control outputs
* locomotion status
* execution feedback
* control mode status

The exact output granularity can be finalized later.

Possible output types include:

* joint targets
* velocity targets
* torque commands
* base motion commands

---

### Communication

Locomotion communicates only with the Core Communication Layer through gRPC.

---

## 2.7 Logger / Replay / Evaluation

### Role

Logger / Replay / Evaluation is responsible for logging, replay, and evaluation.

It does not participate in the real-time main control loop, but it is important
for research, debugging, and experiment comparison.

It is responsible for:

* saving system events and traces
* saving module input and output summaries
* saving ground truth
* supporting episode replay
* supporting offline evaluation and experiment comparison

---

### Receives

Logger / Replay / Evaluation may receive:

* system events
* module status
* main pipeline I/O summaries
* target estimation results
* decision results
* path planning results
* locomotion outputs
* Simulation ground truth
* configuration snapshots
* experiment metadata

---

### Sends

Logger / Replay / Evaluation may send or expose:

* replay data
* evaluation results
* performance statistics
* comparison analysis results
* visualization analysis data

---

### Communication

Logger / Replay / Evaluation should communicate with the main system
asynchronously.

Possible storage/query mechanisms:

* file system
* object storage
* database
* management API for replay and evaluation queries

---

# 3. Deployment and Project Organization

## 3.1 Docker Environment Management

Each major module should eventually be independently Dockerized:

* simulation
* comm_layer
* perception
* decision_making
* navdp_path_planning
* locomotion
* logger

Each module should have:

* Dockerfile
* .dockerignore
* dependency description
* startup script
* health check

Docker images must preserve the same module ownership as local Conda
environments. A container for one module must not be used to run another module.

---

## 3.2 Compose-Based Startup

Recommended Compose files:

* `docker-compose.dev.yml`
* `docker-compose.full.yml`

The goal is not only to make the system runnable, but also to make team startup
consistent.

Compose should standardize:

* network configuration
* ports
* volume mounts
* module dependencies
* startup order

---

## 3.3 Configuration Management

Configuration should be separated from code.

Recommended structure:

```text
configs/
  system/
  simulation/
  perception/
  decision_making/
  navdp/
  locomotion/
  experiments/
```

This keeps environment management and experiment management separate:

* Docker keeps environments consistent
* Config keeps experiments consistent

---

# Summary

The current architecture should be understood as:

```text
Simulation
  ↕ ROS2
Core Communication Layer
  ↕ gRPC
Perception / Decision / NavDP / Locomotion

Logger / Replay / Evaluation runs asynchronously alongside the main loop.
```

The most important current decisions are:

* Simulation is independent infrastructure.
* The Core Communication Layer is the only central system hub.
* Perception, Decision, Planning, and Locomotion remain independent modules.
* Simulation connects to the core layer through ROS2.
* Internal functional modules connect to the core layer primarily through gRPC.
* Locomotion only talks to the core layer.
* NavDP only handles path planning.
* Shared interfaces should be versioned later.
* Docker manages environments, Compose manages startup, Config manages experiments.
* Each module runs only in its own runtime environment.
