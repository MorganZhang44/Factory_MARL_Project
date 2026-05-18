# NavDP Module

`navdp` is the navigation and waypoint-planning module used by the runtime
system. It exposes a small HTTP boundary to `core` and returns a world-frame
waypoint path for each requested robot.

## Runtime Role

The service takes:

- `robot_id`
- robot world-frame state
- a world-frame `subgoal`
- optional RGB / depth payloads for real NavDP inference

and returns:

- one world-frame waypoint list
- planner metadata describing which planner produced the path

## Planner Layers

The current implementation is no longer just a placeholder. It supports three
planner layers behind the same `/plan` endpoint:

1. **real NavDP**
   - uses `vendor/navdp_baseline`
   - requires `sensors.rgb` and `sensors.depth`
   - loads the checkpoint at `checkpoints/navdp-cross-modal.ckpt`

2. **A* fallback**
   - uses the shared obstacle-map helpers from `marl/research`
   - can be selected directly with `--planner astar`
   - is also used automatically when real NavDP fails or returns a near-static path

3. **straight-line fallback**
   - final fallback when both real NavDP and A* fail

## Directory Layout

```text
navdp/
├── navdp_service.py
├── README.md
├── environment.yml
├── Dockerfile
├── checkpoints/
│   └── navdp-cross-modal.ckpt
└── vendor/
    └── navdp_baseline/
```

## Runtime Boundary

Runtime environment:

```bash
conda activate navdp
```

Launch:

```bash
./scripts/launch_navdp.sh
```

Endpoint:

```text
POST http://127.0.0.1:8889/plan
```

Example input:

```json
{
  "robot_id": "agent_1",
  "robot_state": {"position": [0.0, 0.0], "velocity": [0.0, 0.0]},
  "subgoal": [2.0, 1.0],
  "simulation_state": {}
}
```

Example output:

```json
{
  "robot_id": "agent_1",
  "waypoints": [[0.0, 0.0], [0.5, 0.25], [1.0, 0.5], [1.5, 0.75], [2.0, 1.0]],
  "source_module": "navdp",
  "planner": "straight_line_v1"
}
```

## Runtime Dependencies

- `navdp_service.py` is the online service entrypoint
- `checkpoints/navdp-cross-modal.ckpt` is the default real-NavDP checkpoint
- `vendor/navdp_baseline/` provides the baseline model code
- `marl/research/marl/utils/` provides the A* and obstacle-map helpers used by
  the fallback planner
