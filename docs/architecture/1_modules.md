# Modules

## Overview

This document defines all core modules in the system, including their roles, inputs, outputs, and interaction patterns.

All modules:

* communicate **only through the Core Communication Layer**
* do NOT directly call each other
* operate within a **synchronous execution loop (Version 1)**

---

## Module List

* Simulation
* Perception / Target Estimation
* Decision Making
* NavDP Path Planning
* Locomotion
* Core Communication Layer
* Logger / Replay / Evaluation

---

## 1. Simulation

### Role

* Provide the 2D environment
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
* In Version 1, may expose intruder ground truth for development convenience
* Responsible for applying actions at each timestep

---

## 2. Perception / Target Estimation

### Role

* Convert raw simulation state into **target estimate**
* Handle visibility and uncertainty

### Input

* `simulation/state`

### Output

* `perception/target_estimate`

### Notes

* In Version 1:

  * may directly pass through ground truth
* In future:

  * must handle partial observability
  * estimate target state from sensor data

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

## 4. NavDP Path Planning

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

## 5. Locomotion

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

## 6. Core Communication Layer

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

* Runs **in-process (single machine)**
* Acts as a **message dispatcher**
* Messages are delivered within the same timestep

---

### Non-Responsibilities (Important)

The communication layer does NOT:

* perform any domain logic
* modify payload content
* make decisions
* maintain simulation state

---

## 7. Logger / Replay / Evaluation

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
* operates independently

This design ensures:

* modularity
* scalability
* compatibility with industrial architectures
