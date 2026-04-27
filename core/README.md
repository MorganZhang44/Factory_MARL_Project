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

Responsibilities in Version 1:

* subscribe to Simulation ROS2 topics
* keep the latest robot, intruder, camera, and LiDAR state
* expose a non-blocking Core state API for debugging and observability
* run the Core-owned Web Dashboard frontend together with the control layer
* expose one central place where later perception, decision, planning, and
  locomotion modules can be routed

The Core module should not own Simulation logic. Simulation-side publishers live
under `simulation/ros2`.

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
