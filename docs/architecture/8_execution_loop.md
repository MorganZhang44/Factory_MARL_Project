# Execution Loop

## Overview

Defines the logical execution behavior of the system.

The system is:

> **message-driven with a deterministic logical control order**

Physical processes may run in separate module environments. The sequence below
defines control ownership and data dependencies, not Python call order.

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

    core.dispatch()

    # 2. Perception
    perception.update()
    core.dispatch()

    # 3. Decision
    decision.update()
    core.dispatch()

    # 4. Planning
    planning.update()
    core.dispatch()

    # 5. Locomotion
    locomotion.update()
    core.dispatch()

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

* after each publish, Core routes the newest available state
* Version 1 tries to keep a deterministic frame order

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
* no hidden direct calls between modules
* no module running inside another module's environment

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
