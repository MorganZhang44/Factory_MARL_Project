# Core Visualization Requirements

## Positioning

Visualization in the core layer is not a cosmetic extra. It is a runtime observability and debugging surface for the full system.

Its job is to help us answer:

- which modules are alive
- which links are healthy
- which stage of the control loop is currently active
- where data quality or timing is breaking down

## Architectural Role

The dashboard is now treated as a built-in subfunction of the core communication layer rather than a separate standalone module.

That means the core layer is responsible for:

- routing and orchestration
- state mirroring
- runtime visibility
- debug monitoring

## Main Goals

The dashboard should make it easy to inspect:

1. module health
2. message flow and timing
3. current step in the main execution chain
4. spatial state of robots, targets, paths, and subgoals
5. warnings, stale data, and failures

## Recommended Panels

### Module status

At minimum:

- module name
- online/offline state
- latest heartbeat or update time
- basic runtime status such as `idle`, `running`, or `error`

### Communication status

At minimum:

- upstream/downstream pairing
- message rate
- average latency
- timeout state
- latest message arrival

### Execution-chain summary

The main path should be visible as:

```text
Simulation
  -> Perception / Target Estimation
  -> Decision Making
  -> NavDP Path Planning
  -> Locomotion
  -> Simulation
```

For each stage, the dashboard should show a compact view of:

- recent input summary
- recent output summary
- freshness
- validity

### Spatial view

This should present a lightweight situation map containing:

- robot pose
- target estimate
- target prediction
- subgoals from the decision layer
- planned path from NavDP
- current execution state

### Alerts and errors

The dashboard should surface system issues directly instead of forcing the operator to dig through logs:

- module crash
- missing heartbeat
- timeout
- schema mismatch
- invalid data
- broken control path
- missing critical output

## Implementation Guidance

The preferred implementation is a web dashboard with:

- state mirrored inside the core process
- real-time updates over WebSocket
- REST endpoints only for low-frequency queries

The visualization path must remain decoupled from the control loop so that dashboard failures cannot stall runtime execution.

## MVP

A first useful version should include:

- module online/offline status
- link frequency and latency
- target summary
- decision summary
- planning summary
- locomotion summary
- warning list

Spatial overlays and short-horizon history can be layered on after that baseline is stable.
