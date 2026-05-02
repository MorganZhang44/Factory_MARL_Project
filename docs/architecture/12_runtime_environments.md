# Runtime Environments

## Principle

Each module must run only inside its own runtime environment.

No module may depend on another module's Python environment, Conda environment,
or local package path at runtime.

This is a hard architectural rule.

## Current Environment Ownership

| Module | Runtime environment | Notes |
| --- | --- | --- |
| Simulation | `isaaclab51` | Isaac Sim / Isaac Lab only |
| Core Communication Layer | `core` | ROS2 subscribers, state API, orchestration |
| Visualization | `core` | Runs with Core as the Core-owned dashboard frontend |
| ROS2 tooling / bringup | `ros2` | Shared tooling and launch definitions only |
| Perception | `perception` | HTTP adapter on port 8891; dog self-localization + intruder detection + sensor fusion |
| Decision | future `decision` | gRPC client/server boundary |
| Planning / NavDP | `navdp` | Planning service boundary; loads the NavDP point-goal model behind the adapter |
| Locomotion | `locomotion` | Motion-command service boundary; loads the Go2 low-level actor behind the adapter |

Visualization currently shares the `core` environment because it is a Core-owned
observability surface, not an independent algorithm module.

## Runtime Boundary Rules

* Simulation must be launched from `isaaclab51`.
* Core and Visualization must be launched from `core`.
* Perception must be launched from `perception`.
* NavDP must be launched from `navdp`.
* Locomotion must be launched from `locomotion`.
* A module must not be started from another module's environment.
* A module may communicate with another module only through its public interface.
* Python imports across module source trees are not allowed as runtime coupling.
* Shared schemas must move into a versioned shared interface package before they
  are reused by multiple modules.
* Docker images should preserve the same ownership: one image per module
  environment.
* `marl` remains excluded from the shared Docker path until its runtime is
  intentionally containerized.

## Current Launch Split

Simulation:

```bash
./scripts/launch_simulation.sh
```

Core plus Visualization:

```bash
./scripts/launch_core_dashboard.sh
```

Perception:

```bash
./scripts/launch_perception.sh
```

NavDP:

```bash
./scripts/launch_navdp.sh
```

Locomotion:

```bash
./scripts/launch_locomotion.sh
```

Default ports:

```text
Core state API:        http://localhost:8765
Visualization frontend: http://localhost:8770
Perception adapter:    http://localhost:8891
NavDP adapter:         http://localhost:8889
Locomotion adapter:    http://localhost:8890
```

## Simulation Rule

The Simulation module uses Isaac Sim / Isaac Lab directly.

The current entry point loads the imported SLAM scene, adds two Unitree Go2
robots, one humanoid intruder, cameras, and LiDARs:

```text
simulation/standalone/validate_slam_scene.py
```

Temporary mock ROS2 publishers may remain for integration testing, but they are
not the primary Simulation runtime.
