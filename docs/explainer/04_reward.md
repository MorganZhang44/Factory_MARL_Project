# 04 · Reward Design

**Primary file:** `marl/research/marl/rewards/pursuit_reward.py`  
**Primary config:** `marl/research/configs/mappo_config.yaml`

Reward design is the most important part of the research stack because it decides what kind of teamwork the policy actually learns.

## Reward Components

The current reward mixes six pieces:

- progress reward
- pinning reward
- capture reward
- time bonus
- proximity penalty
- step penalty

## Progress Reward

This gives dense reward whenever the two robots, on average, reduce their distance to the intruder. It is the baseline chase incentive.

## Pinning Reward

This adds persistent reward when a robot stays close to the intruder. The purpose is to encourage one robot to commit and pressure the target instead of both robots waiting for a perfect symmetric setup.

## Capture Reward

This is the sparse high-value reward that triggers when both robots enter the capture radius and form a sufficiently wide angle around the target. Better enclosure geometry yields a higher score.

## Time Bonus

Capture reward is paired with a time-efficiency bonus. Faster capture is worth more, which reduces the tendency to stall for an ideal formation.

## Proximity Penalty

This discourages the two robots from collapsing into the same physical region. It helps preserve role separation and makes enclosing behavior easier to learn.

## Step Penalty

A small negative reward per step pushes the policy away from needless delay.

## Why This Combination Works

Together, these terms shape a specific behavior:

- approach the target consistently
- keep one robot engaged at close range
- preserve separation between teammates
- finish the enclosure quickly once the chance appears

That mix was necessary to move the policy from naive chasing toward deliberate cooperative capture.
