# System Overview

## Goal

Build a multi-agent system (2 agents) that can detect, track, and intercept an intruder in a 2D environment.

The system integrates:

* Perception (state estimation)
* Decision Making (MARL-based strategy)
* Path Planning
* Locomotion (execution)

The main challenge is **system integration**, not individual modules.

---

## High-level Pipeline

```
Simulation → Perception → Decision → Planning → Locomotion → Simulation
```

### Description

* **Simulation**

  * Provides environment, robot dynamics, and ground truth
  * Acts as the system runtime

* **Perception / Target Estimation**

  * Converts raw observations into structured target state
  * Handles uncertainty and temporary target loss

* **Decision Making (MARL)**

  * Takes global state and assigns strategy to each agent
  * Outputs subgoals and coordination decisions

* **NavDP Path Planning**

  * Converts subgoals into executable paths / waypoints
  * Handles obstacle avoidance

* **Locomotion**

  * Executes movement commands
  * Translates paths into low-level control

---

## Key Design Principles

### 1. Clear Separation of Responsibilities

Each module has a well-defined role and **does not overlap** with others

---

### 2. Interface-first Design

All modules communicate through **well-defined interfaces**.

Before implementation, we define:

* What data is passed
* Data format
* Required fields (e.g., timestamp, robot_id)

This ensures:

* Parallel development
* Easy module replacement
* Reduced integration issues

---

### 3. Minimal Working System First

The system will be built incrementally:

* Start with simplified components (e.g., fake perception)
* Ensure full pipeline runs end-to-end
* Gradually replace modules with real implementations

---

### 4. Centralized Logical Data Flow

Although modules are conceptually separated, the system initially runs in a **single-machine setup** with direct data passing.

Distributed communication (ROS2 / gRPC) can be added later if needed.

---

### 5. Decision-Centric Architecture

The core intelligence lies in the **Decision Making module (MARL)**.

Other modules:

* Provide inputs (Perception)
* Execute outputs (Planning + Locomotion)

---

## System Scope (Version 1)

### Included

* 2D environment
* 2 agents + 1 intruder
* Basic perception (can be simplified initially)
* MARL-based decision making
* Path planning and execution
* End-to-end simulation loop

### Not Included (initially)

* Real robot deployment
* Complex sensor noise modeling
* Multi-machine distributed system
* High-performance optimization

---

## Expected Outcome

A working system where:

* Agents receive environment state
* Decision module assigns coordinated strategies
* Agents move according to planned paths
* Intruder can be tracked and intercepted

The system should be:

* Modular
* Replaceable (each module can be improved independently)
* Extensible for future upgrades

---

## Summary

This project focuses on:

> Building a **modular, integrated multi-agent system**, where
> **clear interfaces and data flow** enable reliable coordination between perception, decision, and control.

The priority is to **make the system work end-to-end**, then improve individual modules iteratively.
