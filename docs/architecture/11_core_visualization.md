# Core Visualization

## Decision

The primary visualization interface is a Web Dashboard that reads only from the
Core Communication Layer.

This replaces local-only plotting as the main system visualization path.

## Rationale

A Web Dashboard is preferred because it:

* works naturally with Dockerized systems
* supports remote access
* supports live system status and sensor summaries
* is better suited to system-level debugging than relying only on the simulator UI

## Data Flow

The Core Communication Layer maintains an internal state cache / state mirror
and exposes it through a read-only state API.

The dashboard reads from this mirror:

```text
Simulation ROS2 topics
  -> Core subscribers
  -> Core state mirror
  -> Core REST / WebSocket state API
  -> Visualization frontend
  -> Browser dashboard
```

Real-time updates use WebSocket from Core to the browser.

Low-frequency query, filtering, and health checks use REST.

## Control-Loop Isolation

Visualization must not block the main control path.

Rules:

* visualization reads state summaries or mirrored data
* visualization does not subscribe to Simulation ROS2 topics
* visualization is not inserted into the control loop
* dashboard failures must not stop Core control
* slow browser clients must not slow ROS2 subscriptions
* WebSocket / REST handlers must not mutate authoritative control state

## Current Endpoints

Core state API:

```text
http://localhost:8765
```

REST:

```text
GET /health
GET /api/state
```

WebSocket:

```text
ws://localhost:8765/ws
```

Visualization frontend:

```text
http://localhost:8770
```
