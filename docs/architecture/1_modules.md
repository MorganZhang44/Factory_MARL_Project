# Modules

## Overview

This document defines the core modules in the system, including their roles, inputs, outputs, and current implementation scope.

The goal is to clearly specify how information flows through the system and what each module is responsible for.

---

## 1. Simulation

### Role

* Provide the 2D environment
* Manage agents and intruder states
* Generate observations and ground truth
* Control episode lifecycle (reset, step, termination)

### Input

* motion commands from Locomotion
* control signals (reset, pause, resume)

### Output

* sensor observations (or simplified state)
* robot states (position, velocity)
* intruder ground truth
* episode status

### Current Version Notes

* Runs in a simplified 2D environment
* Can provide direct access to ground truth (for early-stage testing)

---

## 2. Perception / Target Estimation

### Role

* Estimate intruder state from observations
* Handle uncertainty and temporary target loss
* Output structured target information

### Input

* sensor observations (camera, lidar, etc. or simplified inputs)
* robot states
* timestamp

### Output

* target position
* target velocity
* confidence score
* visibility flag

### Current Version Notes

* Can initially use ground truth or simplified estimation
* Will be replaced with a more realistic estimation module later

---

## 3. Decision Making

### Role

* Perform mid-level decision making for the team
* Assign strategy and coordination across agents
* Core module for MARL integration

### Input

* target estimation (position, velocity, confidence)
* robot states (all agents)
* map or environment summary
* optional history/context

### Output

* subgoal for each agent (e.g., waypoint or target position)
* role assignment (e.g., chaser, blocker)
* priority or task mode

### Current Version Notes

* Main focus of the project (MARL)
* Can start with simple rule-based logic before integrating MARL

---

## 4. NavDP Path Planning

### Role

* Convert subgoals into executable paths
* Generate waypoint sequences for navigation
* Handle obstacle avoidance in the environment

### Input

* current robot state
* subgoal from Decision Making
* local map or obstacle information

### Output

* path (global or local)
* waypoint sequence
* planning status (optional)

### Current Version Notes

* Focuses only on path planning (no control)
* Can start with simple straight-line or grid-based planning

---

## 5. Locomotion

### Role

* Execute movement based on planned paths
* Convert waypoints into motion commands
* Handle low-level movement logic

### Input

* path or waypoint sequence
* current robot state

### Output

* motion commands (velocity, direction, etc.)
* execution status

### Current Version Notes

* Can initially use simplified motion (direct movement to waypoint)
* No need for complex control at early stage

---

## 6. Logger / Replay / Evaluation

### Role

* Record system events and data
* Support experiment replay
* Provide evaluation metrics

### Input

* system states (robots, intruder)
* perception outputs
* decision outputs
* planning results
* locomotion outputs

### Output

* logs (for debugging)
* replay data
* evaluation results (e.g., success rate, capture time)

### Current Version Notes

* Does not need to be fully implemented in early stage
* Basic logging is sufficient initially

---

## Summary

The system is composed of modular components connected through a clear data flow:

Simulation → Perception → Decision → Planning → Locomotion → Simulation

Each module:

* has a well-defined role
* processes structured inputs
* produces outputs for the next stage

The design prioritizes:

* modularity
* clarity of data flow
* ease of integration
* incremental development
