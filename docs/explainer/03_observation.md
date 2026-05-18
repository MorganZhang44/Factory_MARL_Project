# 03 · Observation Space and LiDAR Features

**Primary files:**

- `marl/research/marl/envs/pursuit_env.py`
- `marl/research/marl/utils/map_utils.py`

## Observation Design

The actor only sees what is placed into the observation vector, so observation design defines the policy's effective perception.

The current observation is **21-dimensional**:

- `agent_id`
- own position
- own velocity
- teammate position
- teammate velocity
- intruder position
- intruder velocity
- 8 ray-cast LiDAR distances

## Why LiDAR Was Added

Without LiDAR-like obstacle context, the policy knew where the target was but not where walls were. That often led to subgoals inside dead ends or behind obstacles. The ray-cast distances give the network a local geometric sense of free space.

## LiDAR Representation

The environment casts 8 rays at fixed directions:

- 0°
- 45°
- 90°
- 135°
- 180°
- 225°
- 270°
- 315°

Each ray returns a normalized distance in `[0, 1]`:

- `0` means an obstacle is very close
- `1` means the ray stayed clear up to the max range

## Sim-to-Real Intuition

The research environment uses geometric ray casting, while the real robot uses actual LiDAR and depth sensing. The exact signal source is different, but the policy-facing idea is the same: obstacle-aware local space cues that help the planner avoid pushing robots into walls.

## Configuration Coupling

The configured `obs_dim` must match the actual observation length. If those drift apart, training or inference will fail when loading weights.
