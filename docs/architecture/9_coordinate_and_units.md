# Coordinate System and Units

## Overview

This document defines all spatial and physical conventions used in the system.

This ensures:

* consistency across modules
* no ambiguity in interpretation

---

## Coordinate System

* Type: **2D Cartesian**
* Frame: `world`
* Origin: **center of the map (0, 0)**
* Authority: matches Isaac Lab world frame in `simulation/scenes/warehouse_scene_cfg.py`

---

## Map Definition

* Map size: **20 × 20 meters**
* X range: **[-10.0, +10.0]**
* Y range: **[-10.0, +10.0]**
* Units: **meters**

---

## Position

```id="position-def"
position = [x, y]
```

* Unit: meters
* Range: x ∈ [-10.0, +10.0], y ∈ [-10.0, +10.0]

---

## Velocity

```id="velocity-def"
velocity = [vx, vy]
```

* Unit: meters / second (m/s)

---

## Time

* Unit: seconds
* Source: simulation clock

---

## Orientation (Optional)

Future extension:

```id="orientation-def"
theta = angle (radians)
```

---

## Agent Size

* radius: **0.25 m** (matching Go2 body footprint in scene)

---

## Static Obstacles (from warehouse_scene_cfg.py)

All positions in world frame (center origin):

| Type | Positions (x, y) |
|------|------------------|
| Perimeter walls | ±10m boundary |
| Interior wall 1 | (-3.0, 3.0), length 6m, horizontal |
| Interior wall 2 | (4.0, -2.5), length 5m, vertical |
| Interior wall 3 | (2.0, -5.0), length 4m, horizontal |
| Pillar 1 | (-5.0, -5.0) |
| Pillar 2 | (5.0, 5.0) |
| Pillar 3 | (-2.0, 7.0) |
| Pillar 4 | (7.0, -7.0) |
| Box 1 | (2.0, 6.0) |
| Box 2 | (-6.0, -2.0) |
| Box 3 | (-3.0, -7.0) |
| Box 4 | (6.5, 2.0) |

---

## Distance Metric

* Euclidean distance

```id="distance"
d = sqrt((x1-x2)^2 + (y1-y2)^2)
```

---

## Consistency Rules

* All modules must use:

  * same coordinate frame
  * same units
* No implicit conversion allowed

---

## Summary

This document defines:

* how space is represented
* how motion is measured
* how all modules stay consistent

It is critical for avoiding integration bugs.
