# Data Flow

## Overview

This document defines how data moves between modules in the system.

The goal is to clearly specify:

* who sends data to whom
* what type of data is transmitted
* the overall direction of the pipeline

This ensures all modules can be developed and integrated consistently.

---

## High-level Data Flow

```id="flow-diagram"
Simulation → Perception → Decision → Planning → Locomotion → Simulation
```

This forms a closed loop:

* the system observes the environment
* makes decisions
* executes actions
* updates the environment

---

## Detailed Data Flow

### 1. Simulation → Perception

**Description**

* Simulation provides observations and system state

**Data**

* sensor observations (or simplified state)
* robot states (position, velocity)
* intruder ground truth (optional for early stage)
* timestamp

---

### 2. Perception → Decision

**Description**

* Perception outputs structured target estimation

**Data**

* target position
* target velocity
* confidence
* visibility flag
* timestamp

---

### 3. Simulation → Decision (Parallel Input)

**Description**

* Decision may also receive global robot state directly

**Data**

* all robot states
* environment/map summary

---

### 4. Decision → Planning

**Description**

* Decision assigns subgoals and strategy

**Data**

* subgoal (target position or waypoint)
* role assignment
* priority / mode

---

### 5. Planning → Locomotion

**Description**

* Planning generates executable paths

**Data**

* path
* waypoint sequence
* planning status (optional)

---

### 6. Locomotion → Simulation

**Description**

* Locomotion sends control commands to simulation

**Data**

* motion commands (velocity, direction, etc.)

---

### 7. Global Logging Flow (Optional)

All modules may send data to Logger:

* perception outputs
* decision outputs
* planning results
* robot states
* episode results

---

## Data Flow Characteristics

### 1. Directionality

* Data flows primarily in a **forward pipeline**
* Control feedback closes the loop via Simulation

---

### 2. Synchronization (Initial Version)

* All modules operate in a **synchronous loop**
* Same timestep shared across modules

---

### 3. Central State Source

* Simulation is the **source of truth** for:

  * robot states
  * environment state

---

### 4. Incremental Complexity

* Early stage:

  * simplified perception (may use ground truth)
* Later stage:

  * replace with real estimation
  * introduce noise and uncertainty

---

## Summary

The system follows a structured and modular data pipeline:

* Each module transforms data into a higher-level representation
* Information flows consistently between modules
* The loop enables continuous perception → decision → action

This design ensures:

* clear integration points
* minimal ambiguity between modules
* support for incremental development
