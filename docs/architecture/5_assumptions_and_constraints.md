# Assumptions and Constraints

## Overview

This document defines the **assumptions** and **constraints** of the system.

The goal is to:

* align team understanding
* avoid hidden inconsistencies
* clearly separate current scope from future extensions

---

## Assumptions

### 1. Environment

* The system operates in a **2D map**
* The map is bounded and known
* Obstacles (if present) are static in Version 1

---

### 2. Agents

* There are exactly **2 agents**
* Agents have:

  * perfect self-localization (position, velocity)
  * full access to their own state
* Agents can move freely within the map

---

### 3. Intruder (Target)

* There is **1 intruder**
* Intruder follows a predefined or controllable policy
* Intruder state (position, velocity) exists at all times (ground truth available in simulation)

---

### 4. Perception (Initial Version)

* Perception can access:

  * ground truth intruder state (for early development)
* No sensor noise in Version 1
* Target visibility can be simplified (always visible or controlled manually)

---

### 5. Communication

* All modules run on a **single machine**
* Communication is:

  * instantaneous
  * lossless
* No delay or packet loss is considered

---

### 6. Time and Execution

* The system runs in a **synchronous loop**
* All modules share the same timestep
* No asynchronous execution in Version 1

---

### 7. Coordinate System

* A global **2D Cartesian coordinate system** is used
* All modules use the same reference frame (`frame_id = world`)

---

## Constraints

### 1. Scope Constraints

* Focus is on **system integration + decision making**
* Not targeting:

  * real robot deployment
  * hardware constraints
  * real-time performance guarantees

---

### 2. Complexity Constraints

* Perception is simplified initially
* Planning can use simple algorithms (e.g., straight-line or basic pathfinding)
* Locomotion can use simplified motion models

---

### 3. Resource Constraints

* The system should run on a **single machine**
* No requirement for distributed computing
* No requirement for GPU beyond MARL training

---

### 4. Development Constraints

* System must support **parallel development by 3 members**
* Interfaces must remain stable once defined
* Modules should be independently testable

---

### 5. Evaluation Constraints

* Performance metrics may include:

  * capture success rate
  * time to intercept
* No requirement for benchmarking against external systems in Version 1

---

## Future Relaxations

The following assumptions may be relaxed in later versions:

* Introduce **sensor noise** and partial observability
* Move to **asynchronous communication** (ROS2 / gRPC)
* Support **multi-machine deployment**
* Extend to **more agents and more complex environments**

---

## Summary

These assumptions and constraints define the **operating boundaries** of Version 1.

They ensure:

* a controlled and manageable system scope
* consistent understanding across team members
* a stable foundation for incremental improvements

Any changes to these assumptions should be **explicitly updated in this document** before implementation.
