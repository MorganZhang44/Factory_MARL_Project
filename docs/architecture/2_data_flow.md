# Data Flow

## Overview

This document defines how data flows through the system using the **Core Communication Layer**.

All communication follows:

```id="pattern"
Module → publish → Communication Layer → route → Module
```

---

## Logical Pipeline

The system follows a conceptual pipeline:

```id="logical-flow"
Simulation → Perception → Decision → Planning → Locomotion → Simulation
```

This represents **data transformation order**, not direct calls.

---

## Topic-Based Data Flow

---

## 1. Simulation → Communication Layer

### Topic: `simulation/state`

**Published Data**

* robot states
* intruder state (Version 1)
* map / obstacle information
* timestamp

---

### Consumed By

* Perception
* Decision
* Planning
* Locomotion
* Logger

---

## 2. Perception → Communication Layer

### Topic: `perception/target_estimate`

**Published Data**

* target position
* target velocity
* confidence
* visibility flag

---

### Consumed By

* Decision
* Logger

---

## 3. Decision → Communication Layer

### Topic: `decision/subgoal`

**Published Data**

* subgoal (absolute position)
* robot_id
* mode
* priority

---

### Consumed By

* Planning
* Logger

---

## 4. Planning → Communication Layer

### Topic: `planning/path`

**Published Data**

* waypoint sequence
* robot_id

---

### Consumed By

* Locomotion
* Logger

---

## 5. Locomotion → Communication Layer

### Topic: `locomotion/motion_command`

**Published Data**

* velocity command (vx, vy)
* robot_id

---

### Consumed By

* Simulation
* Logger

---

## 6. Logger (Global Subscriber)

### Subscribes To

* all topics

---

## Data Flow Properties

---

### 1. Broadcast Semantics

* Messages are broadcast to **all subscribers of a topic**
* No direct addressing required

---

### 2. Single Source of Truth

* `simulation/state` is authoritative
* All modules must rely on it for:

  * robot state
  * environment state

---

### 3. Synchronous Processing (Version 1)

* All modules operate within the same timestep
* Execution order is enforced by the loop
* No asynchronous buffering

---

### 4. Message Overwrite Rule

* For each topic:

  * only the **latest message per timestep is used**
* No historical buffering in Version 1

---

### 5. Deterministic Flow

Within one timestep:

```id="flow-order"
Simulation
  ↓
Perception
  ↓
Decision
  ↓
Planning
  ↓
Locomotion
  ↓
Simulation (next step)
```

---

## Example Flow (One Timestep)

```id="example"
1. Simulation publishes simulation/state
2. Perception consumes → publishes target_estimate
3. Decision consumes → publishes subgoal
4. Planning consumes → publishes path
5. Locomotion consumes → publishes motion_command
6. Simulation applies command
```

---

## Map and Obstacle Flow

* Map and obstacle information are included in:

  * `simulation/state`
* Planning uses this for path generation

---

## Future Extensions

* asynchronous message queues
* multi-rate modules
* distributed communication (ROS2 / gRPC)
* message history buffering

---

## Summary

The system uses a **topic-based, communication-driven data flow**:

* modules are decoupled
* communication is standardized
* execution remains deterministic

This ensures:

* consistent integration
* ease of debugging
* scalability for future systems
