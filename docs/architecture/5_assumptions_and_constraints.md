# Assumptions and Constraints

## Overview

This document defines the **assumptions and constraints** of Version 1 of the system.

The goal is to:

* ensure consistent understanding across modules
* eliminate ambiguity
* define a stable development scope

---

# Assumptions

---

## 1. Environment

* The system operates in an Isaac Sim / Isaac Lab scene
* Planning and high-level reasoning may use a projected **2D Cartesian map**
* Coordinate system:

  * origin at **center (0, 0)**
  * consistent with `9_coordinate_and_units.md`
* Map size:

  * 20 × 20 meters
  * range: x ∈ [-10, 10], y ∈ [-10, 10]

---

## 2. Obstacles

* Obstacles are:

  * static
  * known in advance
* Provided through:

  * `simulation/state → map_info`

---

## 3. Agents

* Exactly **2 agents**
* Each agent has:

  * perfect self-localization
  * full access to its own state
* Agents operate independently but share global information via communication layer

---

## 4. Intruder (Target)

* Single intruder
* Intruder has:

  * position and velocity at all times (ground truth exists in simulation)

---

### Important Clarification

* In Version 1:

  * `intruder_state` MAY be included in `simulation/state`
* However:

  * Decision SHOULD rely on `perception/target_estimate`
* This ensures:

  * future compatibility with partial observability

---

## 5. Perception

* Now an independent module with its own runtime environment (`perception`)
* Current implementation uses:

  * dog self-localization via IMU + LiDAR scan matching
  * intruder detection via CCTV semantic segmentation + LiDAR clustering
  * robust sensor fusion with Kalman tracking
* Ground truth is available for evaluation/logging only
* `visible` flag is derived from sensor detection confidence

---

## 6. Communication

* All modules run on a **single machine**
* Each module runs only in its own runtime environment
* Communication is:

  * synchronous
  * instantaneous
  * lossless

---

## 7. Execution Model

* System runs in a **fixed timestep loop**
* All modules operate within the same timestep
* Message delivery happens within the same loop cycle

---

## 8. Coordinate and Units

* All modules MUST use:

  * same coordinate frame
  * same units
* Defined in:

  * `9_coordinate_and_units.md`

---

# Constraints

---

## 1. Scope Constraints

Focus:

* system integration
* communication architecture
* decision layer (MARL)

Not included:

* real robot deployment
* hardware constraints
* real-time guarantees

---

## 2. Complexity Constraints

* Perception:

  * simplified in Version 1
* Planning:

  * simple algorithms (straight-line / grid)
* Locomotion:

  * simplified velocity control

---

## 3. Communication Constraints

* Communication layer is:

  * lightweight
  * the only routing hub between modules
* No:

  * hidden direct module calls
  * shared Python runtime across modules
  * fault tolerance

---

## 4. Development Constraints

* System must support **parallel development**
* Interfaces must remain stable
* Modules must be independently testable
* A module must not be launched from another module's environment

---

## 5. Evaluation Constraints

Metrics include:

* capture success rate
* time to intercept

No requirement for:

* external benchmarking
* real-world validation

---

# Future Relaxations

Future versions may introduce:

* sensor noise
* partial observability
* asynchronous communication
* multi-machine deployment
* more agents
* dynamic environments

---

# Summary

This document defines the **operating boundaries of Version 1**.

It ensures:

* controlled system complexity
* consistent assumptions across modules
* a stable foundation for incremental development

Any deviation from these assumptions must be:

* explicitly documented
* updated across all relevant modules
