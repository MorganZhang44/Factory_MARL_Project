# Factory MARL Project

This repository contains a modular multi-agent interception system built around
Isaac Sim / Isaac Lab, a Core orchestration layer, and a collection of
service-style modules for perception, planning, decision-making, locomotion,
and real-robot integration.

The main end-to-end pipeline that has been integrated and exercised in practice
is:

```text
Simulation -> Core -> Perception / NavDP / MARL / Locomotion -> Dashboard
```

The primary control loop can be read more narrowly as:

```text
Simulation -> Core -> NavDP -> Locomotion -> Simulation
```

`marl` is already connected to `core` as a runtime service. The default runtime
checkpoint is:

```text
marl/v13_final.pt
```

The dashboard also exposes MARL roles, outputs, and system health.

## Module Layout

Active modules:

- `simulation/`
  - Isaac Sim / Isaac Lab scene runtime
- `core/`
  - orchestration, ROS2 integration, state mirror, and dashboard backend
- `perception/`
  - dog self-localization, intruder detection, and sensor fusion service
- `marl/`
  - online MARL decision runtime plus a separate research tree
- `navdp/`
  - path-planning adapter and real NavDP integration
- `locomotion/`
  - low-level motion adapter and policy runtime
- `ros2/`
  - ROS2 launch assets and shared bringup packaging
- `sim2real/`
  - real-robot Go2 integration subproject

## Documentation

Active documentation now lives under:

- `docs/README.md`
- `docs/architecture/`
- `docs/explainer/`
- `docs/sim2real/`

Recommended starting points:

- [docs/README.md](docs/README.md)
- [docs/architecture/key.md](docs/architecture/key.md)
- [docs/architecture/12_runtime_environments.md](docs/architecture/12_runtime_environments.md)
- [docs/architecture/13_stable_motion_baseline.md](docs/architecture/13_stable_motion_baseline.md)
- [docs/architecture/14_docker_and_github_management_plan.md](docs/architecture/14_docker_and_github_management_plan.md)

## Runtime Ownership

Each module must run only in its own environment:

- `simulation` -> `isaaclab51`
- `core` + dashboard -> `core`
- `perception` -> `perception`
- `marl` -> `marl`
- `navdp` -> `navdp`
- `locomotion` -> `locomotion`
- `sim2real` -> `sim2real`
- `ros2/` remains a tooling / launch layer rather than an independently deployed
  service

This is an architectural rule, not just a convenience.

## Local Startup

The recommended local path is still the `legacy` simulation line.

### 1. Simulation

Environment: `isaaclab51`

```bash
./scripts/launch_simulation.sh --runtime legacy --keep-open
```

Optional intruder motion:

```bash
./scripts/launch_simulation.sh --runtime legacy --keep-open --move-intruder
```

Force CPU if needed:

```bash
./scripts/launch_simulation.sh --device cpu
```

### 2. Perception

Environment: `perception`

```bash
./scripts/launch_perception.sh
```

### 3. NavDP

Environment: `navdp`

```bash
./scripts/launch_navdp.sh
```

For a safer shared baseline:

```bash
NAVDP_DEVICE=cpu ./scripts/launch_navdp.sh
```

### 4. Locomotion

Environment: `locomotion`

```bash
./scripts/launch_locomotion.sh
```

### 5. MARL

Environment: `marl`

```bash
./scripts/launch_marl.sh
```

Default service address:

```text
http://127.0.0.1:8892
```

### 6. Core + Dashboard

Environment: `core`

```bash
./scripts/launch_core_dashboard.sh
```

Dashboard:

```text
http://localhost:8770
```

Core state API:

```text
http://localhost:8765
```

## Recommended Startup Order

Use six terminals:

1. `./scripts/launch_simulation.sh --runtime legacy --keep-open`
2. `./scripts/launch_perception.sh`
3. `./scripts/launch_navdp.sh`
4. `./scripts/launch_locomotion.sh`
5. `./scripts/launch_marl.sh`
6. `./scripts/launch_core_dashboard.sh`

Then open:

```text
http://localhost:8770
```

## Docker

The repository also supports a headless shared-service path through
`compose.yaml`.

Services currently covered by Docker:

- `perception`
- `core`
- `navdp`
- `locomotion`
- `marl`
- `simulation` (headless Isaac Sim baseline)

Examples:

```bash
docker compose up --build perception core navdp locomotion marl
```

```bash
docker compose up --build
```

```bash
docker compose down
```

The current recommendation remains:

- use local module environments for day-to-day development
- use Docker for shared reproduction and headless regression

## Key Entry Files

- `simulation/standalone/validate_slam_scene.py`
- `core/factory_core/control_node.py`
- `core/factory_core/state_mirror.py`
- `core/factory_core/visualization_node.py`
- `perception/perception_service.py`
- `navdp/navdp_service.py`
- `locomotion/locomotion_service.py`
- `marl/marl_service.py`
- `sim2real/scripts/launch_dashboard.sh`

## Current Status

The repository is currently organized around:

- a stable local end-to-end simulation pipeline
- a cleaned module boundary between runtime services
- a separate `marl/research/` tree for training and offline evaluation
- a separate `sim2real/` subproject for Go2 real-robot integration

If you are joining the project fresh, the best onboarding route is:

1. read `docs/README.md`
2. bring up the local simulation chain
3. inspect the dashboard and module contracts
4. only then move to Docker sharing or `sim2real`
