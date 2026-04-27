# ROS2 Module

This module owns ROS2 tooling and bringup definitions.

It must run only in the `ros2` environment when used directly.

The source packages stay with the module that owns their logic:

* `core/ros2/factory_core`: Core Communication Layer ROS2 node
* `simulation/ros2/factory_sim_bridge`: Simulation-side ROS2 bridge
* `ros2/factory_bringup`: launch files and integration bringup

Create the ROS2 conda environment:

```bash
conda env create -f ros2/environment.yml
```

The primary local startup is split by module environment:

```bash
./scripts/launch_simulation.sh
./scripts/launch_core_dashboard.sh
```

Temporary all-in-one or mock integration launchers may exist for interface
testing, but they are not the primary runtime path.

The launch files under `ros2/factory_bringup` should not contain module logic.
They only compose module-owned packages.
