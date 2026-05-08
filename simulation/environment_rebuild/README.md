# Environment Rebuild

This is the fresh environment line intended to replace long-term patching of
the current `legacy` runtime.

## Phase 0 goals

The first goal is to establish a clean simulation base with explicit runtime
decisions:

* one runtime-owned physics scene
* referenced SLAM `PhysicsScene` explicitly disabled
* floor physics material explicitly bound
* robot rigid-body physics material explicitly bound
* dynamic actors spawned programmatically

This line does **not** try to be feature-complete immediately.

It is the clean base for rebuilding:

* sensors
* metadata publishing
* ROS2 bridge
* contact diagnostics

## Entry point

```bash
./scripts/launch_simulation.sh --runtime rebuild
```

Current entrypoint:

* `simulation/standalone/run_environment_rebuild.py`

## Current scope

At this stage, the rebuild runtime is mainly a scene-and-physics smoke-test
entrypoint. It is meant to validate:

* scene loading
* actor spawn
* contact/material assumptions
* single-physics-scene ownership

The next step after this base is stable is to reconnect cameras, LiDAR, and the
current project ROS2 contract one layer at a time.

