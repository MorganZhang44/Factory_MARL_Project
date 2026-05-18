# 02 · Simulation Environment and Episode Logic

**Primary file:** `marl/research/marl/envs/pursuit_env.py`

## Role of the Environment

`PursuitEnv` is the research-side environment used for MARL training and evaluation. It is responsible for:

1. storing robot and intruder state
2. applying robot actions every step
3. updating the intruder with a stochastic motion model
4. computing reward
5. deciding when an episode terminates

## Step Function

At a high level, `step(actions)` does four things:

1. converts the actor output into world-frame subgoals
2. uses A* to move each robot toward its subgoal
3. advances the intruder
4. computes reward and termination flags

The actor does not directly command wheel or joint motion. Instead, it proposes a tactical offset, and classical path following moves the robot through the map.

## Intruder Motion

The intruder follows a noisy random-walk process:

- previous velocity plus random perturbation
- normalized back to a fixed speed
- simple bounce behavior when it hits obstacles

This keeps the target nontrivial but still learnable.

## Capture Rule

Capture is intentionally cooperative. A single robot touching the intruder is not enough.

The current rule requires:

- both robots inside the capture radius
- the angle formed around the intruder to be wide enough

In practice this forces the learned policy to create a two-sided enclosure instead of learning a one-robot chase policy.

## Map and Obstacles

Obstacle geometry is defined in `marl/research/marl/utils/map_utils.py`. Static walls, boxes, and pillars are projected into a binary grid map that is then used by the A* planner.

## Why A* Is Still in the Loop

The research policy decides **where** a robot should go next, not **how** to avoid every obstacle. A* handles obstacle-aware motion between the current pose and the chosen subgoal. This keeps the learning problem tactical rather than low-level geometric.
