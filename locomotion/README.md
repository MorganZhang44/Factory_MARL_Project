# Locomotion Module

This module owns the motion-command interface. It loads the Unitree Go2 flat
velocity actor and returns a 12D low-level joint action when Simulation provides
the matching 48D policy observation.

Runtime environment:

```bash
conda activate locomotion
```

Launch:

```bash
./scripts/launch_locomotion.sh
```

V1 endpoint:

```text
POST http://127.0.0.1:8890/command
```

Input:

```json
{
  "robot_id": "agent_1",
  "robot_state": {"position": [0.0, 0.0], "velocity": [0.0, 0.0]},
  "path": {"waypoints": [[0.0, 0.0], [1.0, 0.0]]}
}
```

Output:

```json
{
  "robot_id": "agent_1",
  "velocity": [0.8, 0.0],
  "action": [0.0, 0.0, 0.0],
  "action_scale": 0.25
}
```

`velocity` is kept as a fallback/debug command. `action` is the learned policy
output and should be applied in Simulation as:

```text
joint_position_target = default_joint_position + action_scale * action
```
