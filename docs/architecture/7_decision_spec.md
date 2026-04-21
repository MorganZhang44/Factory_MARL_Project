# Decision Specification

## Overview

This document defines the Decision module in an **implementation-ready form**.

The Decision module:

* subscribes to system state and perception output
* produces subgoals for each agent
* can be implemented as rule-based or MARL

---

# Inputs (Subscriptions)

Decision subscribes to:

* `simulation/state`
* `perception/target_estimate`

---

# Observation Space

## Per-Agent Observation

```text id="obs"
[
  self_position (2),
  self_velocity (2),

  teammate_position (2),
  teammate_velocity (2),

  target_position (2),
  target_velocity (2)
]
```

---

## Dimension

* Total: **12D**

---

## Target Visibility Handling

If:

```text
visible = false
```

Then:

* use **last known target state**
* optionally apply simple prediction:

  * constant velocity model

---

# Action Space

## Definition

```text id="action"
action = [x, y]
```

---

## Meaning

* absolute world coordinate (NOT relative)
* interpreted as **subgoal**

---

## Output Mapping

Each agent publishes:

```json id="output"
{
  "robot_id": "agent_i",
  "subgoal": [x, y],
  "mode": "intercept",
  "priority": 1
}
```

---

# Reward Function

## Core Form

```text id="reward"
R = 
  - w1 * distance_to_target
  + w2 * capture_bonus
  + w3 * team_spread_bonus
  - w4 * collision_penalty
```

---

## Components

### Distance Reward

* encourages moving toward target

---

### Capture Bonus

Capture occurs when:

```text id="capture"
distance(agent, target) ≤ 0.5 m
```

Reward:

* large positive value

---

### Team Spread

Encourage:

* agents not collapsing into same position

---

### Collision Penalty

Penalty if:

* agents collide
* or invalid motion

---

# Episode Termination

Episode ends when ANY condition is met:

```text id="termination"
1. capture achieved
2. max timestep reached
3. agent leaves map boundary
```

---

# Training Paradigm

## CTDE (Centralized Training, Decentralized Execution)

---

### Training

* global state available:

  * both agents
  * target state

---

### Execution

* each agent uses local observation
* no direct agent communication

---

# Mode Field

## Definition

```text id="mode"
intercept
chase
search
idle
```

---

## Usage

Version 1:

* can be fixed = "intercept"

Future:

* may be predicted by policy

---

# Initial Implementation

Version 1:

* rule-based decision:

  * subgoal = target position
* no MARL yet

---

# Consistency Requirements

Decision MUST:

* use only subscribed topics
* output exactly one subgoal per agent per timestep
* use absolute coordinates
* follow schema in `3_interfaces.md`

---

# Summary

Decision module:

* maps observation → subgoal
* defines agent coordination behavior
* is the only learning-based component (future)

It is designed to:

* be simple initially
* be replaceable with MARL
* integrate seamlessly with the communication layer
