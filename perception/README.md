# Perception

`perception/` is an independent HTTP service module in the current project.

It does not launch Isaac Sim directly and does not read scenes on its own.
Instead, it consumes requests packaged by `core` and exposes:

- `GET /health`
- `POST /estimate`

Entry point:

- `perception/perception_service.py`

## Structure

The module is organized into three layers:

- `perception_service.py`
  - online HTTP adapter
- `perception/perception/`
  - perception algorithm core
- `environment/`
  - static scene geometry and runtime data structures

## What Is Kept

The repository keeps the parts that are actually used by the current runtime:

- `perception_service.py`
  - the project’s HTTP integration layer
- `perception/perception/pipeline.py`
  - main perception pipeline assembly
- `perception/perception/camera_detector.py`
  - camera-side detection
- `perception/perception/lidar_detector.py`
  - LiDAR-side detection
- `perception/perception/dog_localizer.py`
  - robot self-localization logic
- `perception/perception/fusion.py`
  - multi-source fusion
- `perception/perception/scan_matching.py`
  - scan / point-cloud matching
- `perception/perception/transforms.py`
  - coordinate transforms
- `perception/perception/types.py`
  - internal perception data types
- `environment/static_scene_geometry.py`
  - static geometry helpers
- `environment/types.py`
  - runtime-facing data structures
- `environment.yml`
  - dedicated conda environment
- `Dockerfile`
  - Docker image definition

Older side experiments such as separate pose servers, visualization scaffolds,
or unrelated migration leftovers are no longer part of the maintained runtime
path.

## Local Startup

```bash
./scripts/launch_perception.sh
```

Default environment:

```text
perception
```

## Docker Startup

```bash
docker compose up --build perception
```

## Runtime Boundary

The recommended data flow is:

```text
simulation -> core -> perception
```

Meaning:

- `simulation` publishes raw sensors and state
- `core` mirrors, packages, throttles, and calls the HTTP adapter
- `perception` is responsible only for state estimation output

`perception` does not subscribe to ROS2 topics directly and does not read Isaac
Sim directly. It relies on `core` to convert runtime state into
`EnvironmentSensorFrame` payloads before handing them to the perception core.

That boundary is intentional:

- `simulation` owns raw data production
- `core` owns orchestration and state mirroring
- `perception` owns estimation logic
