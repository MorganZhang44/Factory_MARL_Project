# Execution Loop

## Overview

Defines the exact execution behavior of the system.

The system is:

> **synchronous, timestep-based, message-driven**

---

# Execution Model

Each timestep follows:

```text id="flow"
publish → dispatch → update → publish → dispatch → ...
```

---

# Main Loop

```python id="loop"
while not done:

    # 1. Simulation publishes current state
    simulation.publish_state()

    comm.dispatch()

    # 2. Perception
    perception.update()
    comm.dispatch()

    # 3. Decision
    decision.update()
    comm.dispatch()

    # 4. Planning
    planning.update()
    comm.dispatch()

    # 5. Locomotion
    locomotion.update()
    comm.dispatch()

    # 6. Simulation applies commands
    simulation.apply_motion_commands()
```

---

# Execution Order (Strict)

```text id="order"
1. Simulation
2. Perception
3. Decision
4. Planning
5. Locomotion
5. Simulation (apply)
```

Order MUST NOT change.

---

# Dispatch Rules

---

## 1. Immediate Delivery

* after each publish, `dispatch()` is called
* messages are delivered within same timestep

---

## 2. Topic-Based Routing

* messages routed by topic
* all subscribers receive message

---

## 3. Overwrite Rule

* per topic:

  * only latest message is kept
* no message history

---

## 4. One Message per Topic per Module

* each module publishes at most:

  * 1 message per topic per timestep

---

# Message Consumption Rule

Each module:

* reads latest message from each subscribed topic
* processes once per timestep
* does NOT reprocess same message

---

# Simulation Step Rule

Important:

* Simulation does NOT step automatically
* It only:

  * publishes state
  * then later applies action

---

## Timeline

```text id="timeline"
t:
  publish state

t:
  compute action

t:
  apply action

t+1:
  new state
```

---

# Deterministic Behavior

System MUST ensure:

* same input → same output
* no race conditions
* no async execution

---

# Minimal Version Behavior

Simplifications allowed:

* perception = ground truth
* decision = direct chase
* planning = straight line
* locomotion = direct velocity

---

# Debugging Rules

Check in order:

1. topic publishing
2. topic subscription
3. message content
4. execution order
5. loop closure

---

# Summary

Execution loop ensures:

* strict ordering
* deterministic behavior
* correct data flow

It is the **runtime backbone of the system**.
