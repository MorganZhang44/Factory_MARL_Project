# Integration Plan

## Overview

This document defines how the system will be integrated step by step.

The strategy:

> **Build communication backbone first, then plug modules into it incrementally**

---

# Guiding Principles

---

## 1. Communication First

* Build Core Communication Layer before modules
* All modules MUST integrate through it

---

## 2. End-to-End Early

* System must run end-to-end as early as possible
* even with simplified modules

---

## 3. Stable Interfaces

* Interfaces in `3_interfaces.md` are fixed contracts
* changes must be coordinated

---

## 4. Always Runnable

* system must run at every stage
* no broken intermediate states

---

# Integration Stages

---

## Stage 1: Communication Layer Skeleton

### Tasks

* implement publish / subscribe
* implement topic routing
* implement message envelope

---

### Done When

* one module can publish a message
* another module receives it via topic
* message structure preserved
* routing is deterministic

---

## Stage 2: Simulation Integration

### Tasks

* publish `simulation/state`
* subscribe to `locomotion/motion_command`

---

### Done When

* simulation publishes state every timestep
* simulation correctly applies motion commands
* state updates reflect actions

---

## Stage 3: Perception Integration

### Tasks

* subscribe to `simulation/state`
* publish `perception/target_estimate`

---

### Done When

* target estimate is generated each timestep
* decision can consume it
* confidence / visible fields valid

---

## Stage 4: Decision Integration

### Tasks

* subscribe to:

  * `simulation/state`
  * `perception/target_estimate`
* publish `decision/subgoal`

---

### Done When

* one subgoal per agent per timestep
* subgoal is valid coordinate
* downstream modules can consume it

---

## Stage 5: Planning Integration

### Tasks

* subscribe to `decision/subgoal`
* subscribe to `simulation/state`
* publish `planning/path`

---

### Done When

* valid waypoint sequence generated
* no empty paths
* respects obstacles

---

## Stage 6: Locomotion Integration

### Tasks

* subscribe to `planning/path`
* publish `locomotion/motion_command`

---

### Done When

* agents move according to path
* movement is stable
* closed loop is achieved

---

## Stage 7: MARL Integration

### Tasks

* replace rule-based decision
* integrate policy
* ensure interface unchanged

---

### Done When

* MARL outputs valid subgoals
* no change required in planning / locomotion
* system runs end-to-end

---

## Stage 8: Logging and Evaluation

### Tasks

* subscribe Logger to all topics
* record system data

---

### Done When

* logs are complete
* replay possible
* metrics computed

---

# Integration Order

```text id="order"
Communication → Simulation → Perception → Decision → Planning → Locomotion → MARL → Evaluation
```

---

# Parallel Development Strategy

---

## Member Split

* Member A → Perception
* Member B → Decision (MARL)
* Member C → Planning + Locomotion

---

## Shared Responsibility

* communication layer
* interfaces
* integration testing

---

# Integration Checkpoints

---

1. Communication works
2. Simulation runs
3. Perception produces estimates
4. Decision outputs subgoals
5. Planning outputs paths
6. Locomotion closes loop
7. MARL replaces logic

---

# Definition of Integrated System

System is integrated when:

* all modules use communication layer
* no direct calls exist
* full loop runs
* outputs are valid and consistent

---

# Summary

This plan ensures:

* controlled system growth
* continuous functionality
* minimal integration risk

The key idea:

> **Build small, validate early, integrate continuously**
