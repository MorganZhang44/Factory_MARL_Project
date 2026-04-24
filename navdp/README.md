# NavDP Module

This module owns the planning interface for the current integration chain.

Runtime environment:

```bash
conda activate navdp
```

Launch:

```bash
./scripts/launch_navdp.sh
```

V1 endpoint:

```text
POST http://127.0.0.1:8889/plan
```

Input:

```json
{
  "robot_id": "agent_1",
  "robot_state": {"position": [0.0, 0.0], "velocity": [0.0, 0.0]},
  "subgoal": [2.0, 1.0],
  "simulation_state": {}
}
```

Output:

```json
{
  "robot_id": "agent_1",
  "waypoints": [[0.0, 0.0], [0.5, 0.25], [1.0, 0.5], [1.5, 0.75], [2.0, 1.0]]
}
```

The current implementation is a clean adapter placeholder. It preserves the
module boundary and can be replaced by the real NavDP model without changing
Core.
