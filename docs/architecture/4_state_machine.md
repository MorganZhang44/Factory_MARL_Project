# State Machine

## Overview

This document defines the **high-level behavioral states** of the system.

It provides:

* structured behavior transitions
* interpretable system modes
* support for decision logic design

---

## Implementation Role

In Version 1:

> The state machine is used as a **conceptual and evaluation layer**, not a hard controller.

---

### Meaning

* MARL policy produces actions directly
* state machine is used to:

  * interpret behavior
  * analyze system performance
  * optionally provide features to policy

---

## Core States

---

### 1. Patrol

* no target detected
* agents explore environment

---

### 2. Detect

* target becomes visible

---

### 3. Chase

* target visible and tracked
* agents move toward target

---

### 4. Intercept

* agents attempt coordinated interception

---

### 5. Lost Target

* target no longer visible

---

### 6. Search

* agents attempt to re-detect target

---

## Transition Conditions

---

### Parameters

```text
confidence_threshold = 0.7
intercept_distance_threshold = 1.0 m
lost_target_timeout = 2.0 s
search_timeout = 5.0 s
```

---

### Transitions

**Patrol → Detect**

* visible = true

---

**Detect → Chase**

* confidence ≥ confidence_threshold

---

**Chase → Intercept**

* distance_to_target ≤ intercept_distance_threshold

---

**Chase → Lost Target**

* visible = false

---

**Lost Target → Search**

* time_since_last_seen ≥ lost_target_timeout

---

**Search → Patrol**

* search duration ≥ search_timeout

---

**Lost Target → Chase**

* visible = true

---

## State Variables

* current_state
* time_in_state
* last_seen_position
* last_seen_timestamp

---

## Relationship with MARL

---

### Option Used (Version 1)

* state is NOT enforced externally
* MARL learns behavior implicitly

---

### Optional Usage

State can be used as:

* additional input feature
* evaluation metric
* debugging tool

---

## Design Notes

* states describe **behavioral phases**
* not all policies must explicitly use them
* useful for:

  * reward shaping
  * debugging
  * visualization

---

## Summary

The state machine:

* structures system behavior conceptually
* provides interpretable transitions
* complements MARL without restricting it

It ensures the system is:

* understandable
* debuggable
* analyzable
