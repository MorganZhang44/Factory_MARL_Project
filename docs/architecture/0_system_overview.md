# System Overview

## Goal

Build a modular multi-agent system (2 agents) that can detect, track, and
intercept an intruder in an Isaac Sim environment.

The system integrates:

* Perception (state estimation)
* Decision Making (MARL-based strategy)
* Path Planning
* Locomotion (execution)

The primary challenge is **system integration across modules**, rather than individual algorithm design.

---

## Design Philosophy

The system is designed not only for academic prototyping, but also with **future industrial integration in mind**.

Key priorities:

* clear module separation
* strict runtime environment ownership
* standardized interfaces
* replaceable components
* scalable communication structure

To achieve this, all modules communicate through a **Core Communication Layer**.

---

## High-level Architecture

```id="arch-overview"
                ┌──────────────────────────┐
                │ Core Communication Layer │
                └───────────┬──────────────┘
                            │
     ┌────────────┬─────────┼─────────┬────────────┐
     │            │         │         │            │
Simulation   Perception  Decision  Planning   Locomotion
```

### Key Idea

* Modules do **not directly call each other**
* All interactions go through the communication layer
* The communication layer handles:

  * message passing
  * routing
  * interface standardization

---

## Functional Pipeline (Logical Flow)

Although physically decoupled, the system follows this logical flow:

```id="logical-flow"
Simulation → Perception → Decision → Planning → Locomotion → Simulation
```

### Description

* **Simulation**

  * Generates environment and system state

* **Perception**

  * Estimates target state from observations

* **Decision (MARL)**

  * Produces coordinated strategy (subgoals)

* **Planning**

  * Converts subgoals into executable paths

* **Locomotion**

  * Executes motion commands

---

## Core Communication Layer

### Role

The Core Communication Layer acts as the **central middleware** of the system.

### Responsibilities

* message routing between modules
* enforcing interface contracts
* decoupling modules
* enabling future distributed deployment

### Design Principle

* **Thin layer (initial version)**
* Only handles communication, not business logic

---

## System Characteristics

### 1. Modular

Each component can be:

* developed independently
* replaced without breaking the system

---

### 2. Interface-driven

All interactions are defined through:

* structured messages
* standardized formats

---

### 3. Incremental Development

The system is built in stages:

* minimal working pipeline first
* progressively replace simplified modules

---

### 4. Single-Machine Execution (Initial)

* All modules run locally
* Communication is abstracted (not physically distributed yet)
* Each module still runs only in its own environment

Current local environment ownership:

* Simulation runs in `isaaclab51`
* Core Communication Layer and Visualization run in `core`
* Perception runs in `perception`
* NavDP runs in `navdp`
* Locomotion runs in `locomotion`
* ROS2 tooling / bringup runs in `ros2`

---

## System Scope (Version 1)

### Included

* Isaac Sim / Isaac Lab simulation scene
* 2 agents + 1 intruder
* perception module with dog self-localization and intruder detection
* MARL decision module (initial rule-based, MAPPO training in progress)
* path planning and locomotion
* communication layer abstraction

---

### Not Included (Initial)

* real robot deployment
* distributed multi-machine system
* advanced fault tolerance
* high-performance optimization

---

## Expected Outcome

A working system where:

* modules communicate through a unified layer
* agents coordinate to intercept the intruder
* the full pipeline runs end-to-end

The system should be:

* modular
* extensible
* aligned with industrial system design practices

---

## Summary

This project focuses on building a:

> **modular, communication-driven multi-agent system**

where:

* modules are loosely coupled
* interfaces are strictly defined
* the system can evolve from a prototype to an industrial-grade architecture

The priority remains:

> **Make the system run end-to-end, while maintaining clean architecture for future scalability.**
