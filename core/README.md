# Core Module

The Core module owns the central control and communication layer.

It must run only in the `core` environment.

Create the Core conda environment:

```bash
conda env create -f core/environment.yml
```

Current ROS2-facing package:

```text
core/ros2/factory_core
```

Current Python source package:

```text
core/factory_core
```

Responsibilities in Version 1:

* subscribe to Simulation ROS2 topics
* keep the latest robot, intruder, camera, and LiDAR state
* expose a non-blocking Core state API for debugging and observability
* run the Core-owned Web Dashboard frontend together with the control layer
* expose one central place where later perception, decision, planning, and
  locomotion modules can be routed
* mirror MARL runtime inputs/outputs for observability without forcing MARL
  into the active control loop yet

The Core module should not own Simulation logic. Simulation-side publishers live
under `simulation/ros2`.

## Directory layout

The module is now intentionally split into three layers:

### 1. Core source code

```text
core/factory_core
```

This directory contains the actual Python implementation:

* `control_node.py`: main Core control node
* `state_mirror.py`: state mirror used by the API and dashboard
* `visualization_node.py`: dashboard frontend node

### 2. Core ROS2 package wrapper

```text
core/ros2/factory_core
```

This directory contains the ROS2 package metadata and packaging files:

* `package.xml`
* `setup.py`
* `setup.cfg`
* `resource/factory_core`

It exists so the Python source in `core/factory_core` can still be exposed as
the ROS2 package `factory_core`.

### 3. Project-level ROS2 bringup

```text
ros2/factory_bringup
```

This directory is outside the module and belongs to the project-level ROS2
integration layer. It contains launch files that start Core together with the
rest of the project.

Run Core and Visualization together:

```bash
./scripts/launch_core_dashboard.sh
```

Open the dashboard in a browser:

```text
http://localhost:8770
```

The Core state API is available at:

```text
http://localhost:8765
```

The dashboard reads the Core-owned state API through WebSocket. It must not
subscribe to Simulation ROS2 topics and must not be inserted into the control
loop.

## Build path

`launch_core_dashboard.sh` builds a ROS2 workspace using:

* `core/ros2`
* `ros2`

This means:

* `core/ros2` provides the `factory_core` ROS2 package
* `ros2` provides the project-level bringup package and launch files

Current dashboard pages include:

* `WorldState`
* `Robot`
* `Perception`
* `MARL`
* `NavDP`
* `Locomotion`
