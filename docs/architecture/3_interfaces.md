# Interfaces

## Overview

This document defines the **strict communication contracts** between modules.

All communication:

* uses **topic-based publish–subscribe**
* follows a **standard message envelope**
* must comply with **schema + semantic rules**

---

# 1. Message Envelope

All messages MUST follow this format:

```json
{
  "message_id": "uuid",
  "timestamp": float,
  "topic": "string",
  "source_module": "string",
  "payload": {}
}
```

---

## Field Definitions

| Field         | Required | Description               |
| ------------- | -------- | ------------------------- |
| message_id    | YES      | unique id                 |
| timestamp     | YES      | simulation time (seconds) |
| topic         | YES      | topic name                |
| source_module | YES      | publisher module          |
| payload       | YES      | message content           |

---

## Rules

* `target_module` is NOT used (broadcast model)
* routing is based on **topic only**
* payload MUST NOT be modified by communication layer

---

# 2. Topic Definitions

```text
simulation/state
perception/target_estimate
decision/subgoal
planning/path
locomotion/motion_command
```

---

# 3. Message Schemas

---

## 3.1 Simulation State

### Topic

`simulation/state`

### Schema

```json
{
  "robot_states": [
    {
      "robot_id": "agent_1",
      "position": [x, y],
      "velocity": [vx, vy]
    }
  ],
  "intruder_state": {
    "position": [x, y],
    "velocity": [vx, vy]
  },
  "map_info": {
    "obstacles": []
  }
}
```

---

### Required Fields

* robot_states
* intruder_state (Version 1 only)
* map_info

---

### Notes

* In future versions:

  * `intruder_state` may NOT be available to Decision
* map_info is static in Version 1

---

## 3.2 Target Estimate

### Topic

`perception/target_estimate`

```json
{
  "target_id": "intruder_1",
  "position": [x, y],
  "velocity": [vx, vy],
  "confidence": float,
  "visible": bool
}
```

---

### Required Fields

* position
* velocity
* confidence
* visible

---

### Semantic Rules

* if `visible = false`:

  * position = last known or predicted
  * confidence < threshold

---

## 3.3 Decision Output

### Topic

`decision/subgoal`

```json
{
  "robot_id": "agent_1",
  "subgoal": [x, y],
  "mode": "intercept",
  "priority": int
}
```

---

### Required Fields

* robot_id
* subgoal

---

### Enum: mode

```text
intercept
chase
search
idle
```

---

### Semantic Rules

* subgoal is **absolute world coordinate**
* one message per robot per timestep

---

## 3.4 Planning Output

### Topic

`planning/path`

```json
{
  "robot_id": "agent_1",
  "waypoints": [
    [x1, y1],
    [x2, y2]
  ]
}
```

---

### Required Fields

* robot_id
* waypoints

---

### Rules

* waypoints MUST NOT be empty
* first waypoint should be near current position

---

## 3.5 Motion Command

### Topic

`locomotion/motion_command`

```json
{
  "robot_id": "agent_1",
  "velocity": [vx, vy]
}
```

---

### Required Fields

* robot_id
* velocity

---

### Rules

* velocity is in **world frame**
* units: m/s

---

# 4. Subscription Table

| Module     | Subscribes To                                |
| ---------- | -------------------------------------------- |
| Perception | simulation/state                             |
| Decision   | simulation/state, perception/target_estimate |
| Planning   | decision/subgoal, simulation/state           |
| Locomotion | planning/path, simulation/state              |
| Simulation | locomotion/motion_command                    |
| Logger     | all topics                                   |

---

# 5. Communication Semantics

---

## Broadcast Model

* all messages are broadcast
* subscribers filter by topic

---

## One Message per Timestep Rule

* each module publishes at most **one message per topic per timestep**

---

## Overwrite Rule

* only the **latest message in a timestep is used**
* no historical buffering

---

## Processing Order

Defined in `8_execution_loop.md`

---

# 6. Validation Rules

Each module MUST ensure:

* payload matches schema
* required fields exist
* values are in valid range
* coordinate frame is consistent

---

## Summary

This interface layer defines:

* strict message format
* unambiguous data contracts
* deterministic communication behavior

It is the **core guarantee of system consistency**.
