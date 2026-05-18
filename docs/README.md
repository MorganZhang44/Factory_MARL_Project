# Documentation Guide

This repository now keeps its active documentation under three branches:

- `docs/architecture/`
  - system architecture
  - runtime contracts
  - ROS2 / Core / Visualization integration notes
- `docs/explainer/`
  - MARL training and environment walkthroughs
  - reward, observation, and policy notes
- `docs/sim2real/`
  - Go2 real-robot connection and ROS2 notes

Recommended reading order:

1. `docs/architecture/0_system_overview.md`
2. `docs/architecture/1_modules.md`
3. `docs/architecture/10_ros2_sim_core_topics.md`
4. `docs/architecture/11_core_visualization.md`
5. `docs/architecture/12_runtime_environments.md`
6. `docs/explainer/README.md`
7. `docs/sim2real/unitree_go2_ros2_guide.md`

Notes:

- The legacy Chinese project-notes tree has been removed.
- The active MAPPO research configuration now lives under:

```text
marl/research/configs/
```

- The active real-robot subproject now lives under:

```text
sim2real/
```
