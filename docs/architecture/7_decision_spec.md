# Decision Specification

## Overview

This document defines the **Decision Making module (MARL)** in a concrete and executable way.

It specifies:

* observation space
* action space
* reward design
* training paradigm

This is the **core intelligence of the system**.

---

## Problem Formulation

* Multi-agent system (2 agents)
* Objective: **intercept intruder**
* Partially observable (future version), fully observable (initial version)

---

## Observation Space

Each agent receives:

```id="obs-space"
[
  self_position (2),
  self_velocity (2),

  teammate_position (2),
  teammate_velocity (2),

  target_position (2),
  target_velocity (2)
]
```

### Dimension

* Total: **12D**

---

## Action Space

### Option A (Recommended – Continuous)

```id="action-space"
action = [x, y]
```

* Represents **subgoal position**
* Passed to Planning module

---

### Option B (Alternative – Discrete)

```id="action-discrete"
action ∈ {chase, intercept_left, intercept_right, search}
```

---

## Reward Function

### Core Reward

```id="reward-core"
R = w1 * (-distance_to_target)
  + w2 * (capture_bonus)
  + w3 * (team_spread_bonus)
  + w4 * (collision_penalty)
```

---

### Components

* **Distance reward**

  * Encourage getting closer to target

* **Capture bonus**

  * Large positive reward when interception succeeds

* **Team coordination**

  * Encourage agents to spread (avoid redundancy)

* **Collision penalty**

  * Penalize collisions or invalid movement

---

## Training Paradigm

### CTDE (Centralized Training, Decentralized Execution)

* Training:

  * agents access **global state**
* Execution:

  * agents act based on local observation

---

## Policy Output

Each agent outputs:

```id="policy-output"
{
  "subgoal": [x, y],
  "mode": "intercept"
}
```

---

## Initial Simplification

Version 1:

* Fully observable environment
* Continuous action (subgoal)
* Simple reward:

  * distance + capture

---

## Future Extensions

* Partial observability
* Communication between agents
* Role-based policies
* Curriculum learning

---

## Summary

The Decision module:

* maps state → subgoal
* learns cooperative interception
* is the **core learning component** of the system
