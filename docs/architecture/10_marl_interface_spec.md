# MARL Integration Interface Specification

## Overview

This document specifies the communication boundary between the **Isaac Sim / Isaac Lab (Perception & Navigation)** modules and the **MARL (Decision)** external Python process. 

To ensure our system components connect seamlessly via the Communication Layer (e.g., ROS2 or Sockets), both teams must adhere to the data structures and conventions defined below.

---

## 1. Frame & Coordinate Conventions

Before sending any data, please ensure:
* **Global Frame**: All positions and velocities must be transformed into the **World Coordinate Frame** (absolute coordinates).
* **Dimensionality**: The MARL engine requires **2D Top-Down Data** `(x, y)`. The Z-axis (height/elevation) and orientations (quaternions/Euler angles) are inherently handled by your locomotion module and should be stripped before sending to MARL.
* **Units**: Metres (m) for positions, Metres/second (m/s) for velocity.

---

## 2. Input to MARL (From Perception & Simulation)

**Topic / Source**: `simulation/state` & `perception/target_estimate`
**Frequency requirement**: ~10 Hz (The macro-level decision rate)

At every tick where MARL is queried for an action, it requires a synchronized snapshot of the current state. 

### Expected JSON Payload Structure:

```json
{
  "timestamp": 1713800000.123,
  "payload": {
    "target": {
      "pos_2d": [2.5, -1.0],  // Target Absolute X, Y (from Perception)
      "vel_2d": [0.5, 0.0]    // Target Velocity vX, vY (from Perception)
    },
    "agents": {
      "agent_1": {
        "pos_2d": [-2.0, -2.0], // Dog 1 Absolute X, Y (from Sim State)
        "vel_2d": [0.0, 0.0]    // Dog 1 Velocity vX, vY (from Sim State)
      },
      "agent_2": {
        "pos_2d": [-2.0, 1.6],  // Dog 2 Absolute X, Y (from Sim State)
        "vel_2d": [0.0, 0.0]    // Dog 2 Velocity vX, vY (from Sim State)
      }
    }
  }
}
```

> **Data Handling Note for Perception**: If the target (intruder) is temporarily occluded or out of LiDAR/camera FOV, the Perception module should either publish the last-known position or pass a state flag. (Currently, the pure MARL model expects continuous coordinates to calculate its encirclement strategy).

---

## 3. Output from MARL (To Navigation)

**Topic / Source**: `decision/subgoal`
**Frequency requirement**: ~10 Hz (Matches input rate)

MARL computes the high-level tactics ("which path to flank the target"). It **does not output motor torques or exact joint velocities**. Instead, it outputs a **Subgoal (Local Waypoint)** for each dog.

### Expected JSON Payload Structure:

```json
{
  "timestamp": 1713800000.123,
  "commands": {
    "agent_1": {
        "subgoal_2d": [-1.0, -1.5]  // Where Dog 1 needs to go immediately 
    },
    "agent_2": {
        "subgoal_2d": [-0.5, 2.0]   // Where Dog 2 needs to go immediately
    }
  }
}
```

---

## 4. Division of Responsibilities (Crucial)

To prevent our algorithms from conflicting with each other, we divide the work mapping exactly to the `Simulation → Perception → Decision → Planning/Locomotion` pipeline:

1. **Strategic Flanking & Target Tracking (MARL)**:
   The MARL model considers the global structure to ensure the two dogs do not overlap and successfully pincer the target. It issues a `subgoal_2d` slightly ahead of the dog.
2. **Local Collision Avoidance & Path Smoothing (Isaac Navigation)**:
   If a dynamic, unmapped obstacle (like a tumbling trash can) appears between the dog and the `subgoal_2d`, the **Navigation module inside Isaac Lab is entirely responsible for local micro-avoidance**. MARL will not issue micro-corrections for immediate sudden collisions.
3. **Execution (Isaac Locomotion)**:
   Converting `subgoal_2d` into leg kinematics / base linear velocities must be handled via the Isaac Locomotion policies/controllers.
