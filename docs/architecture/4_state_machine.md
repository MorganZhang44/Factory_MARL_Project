# State Machine

## Overview

This document defines the **high-level behavior states** of the system and how it transitions between them.

The goal is to:

* structure agent behavior logically
* avoid chaotic decision outputs
* provide a clear framework for MARL policy design

The state machine operates at the **team / system level**, influencing how agents act collectively.

---

## Core States

### 1. Patrol

**Description**

* Default state when no target is detected

**Behavior**

* Agents explore the environment
* Maintain coverage of the map

---

### 2. Detect

**Description**

* Target is observed for the first time

**Behavior**

* Initialize tracking
* Share target information across agents

---

### 3. Chase

**Description**

* Target is visible and being followed

**Behavior**

* Agents move toward predicted target position
* Maintain tracking and reduce distance

---

### 4. Intercept

**Description**

* Agents attempt to block or surround the target

**Behavior**

* Assign roles (e.g., chaser, blocker)
* Move to strategic positions (not just direct pursuit)

---

### 5. Lost Target

**Description**

* Target is no longer visible

**Behavior**

* Use last known position and velocity
* Predict possible target trajectory

---

### 6. Search

**Description**

* Target has been lost for a period of time

**Behavior**

* Agents spread out to re-detect target
* Switch from prediction to exploration

---

## State Transitions

```id="state-flow"
Patrol → Detect → Chase → Intercept
             ↓        ↓
         Lost Target → Search → Patrol
```

---

## Transition Conditions

### Patrol → Detect

* target becomes visible

---

### Detect → Chase

* target confirmed (confidence above threshold)

---

### Chase → Intercept

* distance to target below threshold
* interception condition satisfied

---

### Chase → Lost Target

* target becomes invisible

---

### Lost Target → Search

* target not recovered within time threshold

---

### Search → Patrol

* search timeout reached without detection

---

### Lost Target → Chase

* target re-detected

---

## State Variables (Optional)

The system may maintain:

* `current_state`
* `time_in_state`
* `last_seen_target_position`
* `last_seen_timestamp`

---

## Design Notes

### 1. Relationship with MARL

* State defines the **context** for decision making
* MARL policy can:

  * implicitly learn states
  * or explicitly use state as input

---

### 2. Centralized vs Distributed

* State machine can be:

  * centralized (single shared state)
  * or inferred locally by each agent

Initial version:

* use **shared global state**

---

### 3. Simplification Strategy

Early stage:

* fewer states (e.g., Patrol / Chase / Search)

Later stage:

* refine into full state machine

---

## Summary

The state machine provides:

* structured behavior transitions
* clearer reasoning for decision outputs
* a foundation for MARL training and evaluation

It ensures the system operates in a **controlled and interpretable way**, rather than producing arbitrary actions.
