# Integration Plan

## Overview

This document defines how the system will be built and integrated over time.

The goal is to:

* ensure the system works **end-to-end as early as possible**
* allow parallel development across modules
* reduce integration risk

The strategy is:

> **Build a minimal working pipeline first, then progressively replace components with more advanced implementations**

---

## Guiding Principles

### 1. End-to-End First

* Always prioritize having a **fully connected pipeline**
* Even if modules are simplified or fake

---

### 2. Incremental Replacement

* Start with simple / placeholder implementations
* Replace modules one by one with real versions

---

### 3. Stable Interfaces

* Interfaces defined in `3_interfaces.md` should remain stable
* Changes must be coordinated across modules

---

### 4. Independent Testing

* Each module should be testable in isolation
* Integration happens after basic functionality is verified

---

## Integration Stages

---

## Stage 1: Minimal Pipeline (Baseline System)

### Goal

Get a full loop running:

```id="stage1-flow"
Simulation → Decision → Action → Simulation
```

### Tasks

* Build simple 2D simulation
* Provide direct access to intruder position (no perception module yet)
* Implement simple decision logic:

  * e.g., agents move directly toward intruder
* Implement basic movement (direct velocity control)

### Outcome

* Agents can move and chase intruder
* Full loop runs without errors

---

## Stage 2: Introduce Perception Layer

### Goal

Insert perception into pipeline:

```id="stage2-flow"
Simulation → Perception → Decision → Action → Simulation
```

### Tasks

* Implement Perception module interface
* Initially pass ground truth through perception
* Add basic visibility / confidence logic

### Outcome

* Decision module no longer depends on ground truth directly
* Clear separation between perception and decision

---

## Stage 3: Add Planning Module

### Goal

Insert planning layer:

```id="stage3-flow"
Simulation → Perception → Decision → Planning → Locomotion → Simulation
```

### Tasks

* Implement simple path planning (e.g., straight line or grid-based)
* Convert subgoal into waypoint sequence
* Update locomotion to follow waypoints

### Outcome

* Agents move via planned paths instead of direct commands
* System structure becomes complete

---

## Stage 4: Replace Decision with MARL

### Goal

Integrate MARL into decision layer

### Tasks

* Define observation space based on interfaces
* Define action space (subgoal / role output)
* Implement training loop
* Replace rule-based decision with MARL policy

### Outcome

* Agents exhibit learned cooperative behavior
* Interception strategies improve over time

---

## Stage 5: Improve Realism

### Goal

Make system more realistic and robust

### Tasks

* Add noise or partial observability to perception
* Improve target estimation (handle lost target)
* Refine planning (avoid obstacles, smoother paths)
* Improve locomotion model

### Outcome

* System handles more complex scenarios
* Behavior becomes more robust

---

## Stage 6: Evaluation and Benchmarking

### Goal

Measure system performance

### Tasks

* Define metrics:

  * capture success rate
  * time to intercept
* Run multiple episodes
* Log and analyze results

### Outcome

* Quantitative understanding of system performance
* Comparison between different strategies (e.g., rule-based vs MARL)

---

## Suggested Timeline (Example)

| Week | Focus                            |
| ---- | -------------------------------- |
| 1    | Stage 1 (Minimal pipeline)       |
| 2    | Stage 2 (Perception integration) |
| 3    | Stage 3 (Planning integration)   |
| 4    | Stage 4 (MARL integration)       |
| 5    | Stage 5 (Refinement)             |
| 6    | Stage 6 (Evaluation & report)    |

---

## Responsibilities (Optional)

Each member can focus on:

* Perception / estimation
* Decision (MARL)
* Planning + locomotion + integration

All members should:

* follow interface definitions
* test integration regularly

---

## Summary

This integration plan ensures:

* early system functionality
* controlled increase in complexity
* reduced risk of integration failure

The key idea is:

> **Always keep the system runnable, even if parts are simplified**
