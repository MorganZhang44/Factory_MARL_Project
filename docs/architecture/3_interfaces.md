# Interfaces

## Overview

This document defines the **data interfaces between modules**.

The goal is to standardize:

* data formats
* required fields
* naming conventions

All modules must follow these interfaces to ensure compatibility and parallel development.

---

## Global Conventions

All messages should include:

```json
{
  "timestamp": float,
  "frame_id": "world",
  "robot_id": "agent_1"   // if applicable
}
```

### Notes

* `timestamp`: simulation time (seconds)
* `frame_id`: coordinate frame (default: world)
* `robot_id`: required when referring to a specific agent

---

# 1. Simulation → Perception

### Message: Observation

```json
{
  "timestamp": float,
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
  }
}
```

### Notes

* `intruder_state` can be hidden or noisy in later versions
* Can be simplified in early stage (direct ground truth)

---

# 2. Perception → Decision

### Message: TargetEstimate

```json
{
  "timestamp": float,
  "target_id": "intruder_1",
  "position": [x, y],
  "velocity": [vx, vy],
  "confidence": float,
  "visible": bool
}
```

### Notes

* `visible = false` indicates temporary loss
* `confidence` used for decision robustness

---

# 3. Simulation → Decision

### Message: RobotState

```json
{
  "timestamp": float,
  "robot_states": [
    {
      "robot_id": "agent_1",
      "position": [x, y],
      "velocity": [vx, vy]
    }
  ]
}
```

---

### Message: MapInfo

```json
{
  "map_size": [width, height],
  "obstacles": [
    {
      "type": "rectangle",
      "position": [x, y],
      "size": [w, h]
    }
  ]
}
```

---

# 4. Decision → Planning

### Message: Subgoal

```json
{
  "timestamp": float,
  "robot_id": "agent_1",
  "subgoal": [x, y],
  "mode": "intercept",
  "priority": int
}
```

### Notes

* `mode` examples: "intercept", "block", "search"
* `priority` used if multiple tasks exist

---

# 5. Planning → Locomotion

### Message: Path

```json
{
  "timestamp": float,
  "robot_id": "agent_1",
  "waypoints": [
    [x1, y1],
    [x2, y2]
  ]
}
```

---

# 6. Locomotion → Simulation

### Message: MotionCommand

```json
{
  "timestamp": float,
  "robot_id": "agent_1",
  "velocity": [vx, vy]
}
```

### Notes

* Can be extended to include acceleration or heading
* Simplified in early stage

---

# 7. Logger Interfaces (Optional)

### Message: LogEntry

```json
{
  "timestamp": float,
  "module": "decision",
  "data": {}
}
```

---

# Data Types Summary

| Field      | Type           | Description           |
| ---------- | -------------- | --------------------- |
| position   | [float, float] | 2D coordinates        |
| velocity   | [float, float] | velocity vector       |
| confidence | float          | estimation confidence |
| visible    | bool           | target visibility     |
| waypoints  | list           | sequence of positions |

---

# Version Notes

## Version 1 (Current)

* Simple 2D coordinates
* Perfect or near-perfect perception allowed
* Synchronous loop
* No strict serialization required (can use Python objects)

## Future Versions

* Add noise and uncertainty to perception
* Extend message formats (orientation, acceleration)
* Introduce asynchronous communication (ROS2 / gRPC)

---

## Summary

These interfaces define the **contract between modules**.

They ensure:

* consistency in data exchange
* independence of module implementation
* easier integration and debugging

All modules must adhere to these definitions to maintain system stability.
