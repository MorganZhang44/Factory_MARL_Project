# 16 Environment Rebuild Summary

This note summarizes all environment-related findings gathered while trying to
stabilize the current simulation stack.

The goal is not to keep patching the current `legacy` environment forever.
The goal is to rebuild a cleaner simulation environment with the right
assumptions and the right data contracts from the start.


## 1. Current Situation

There are currently two simulation lines in this repo:

* `legacy`
  * entrypoint:
    [simulation/standalone/validate_slam_scene.py](/home/yyz/projects/Factory_MARL_Project/simulation/standalone/validate_slam_scene.py)
* `rewrite`
  * entrypoint:
    [simulation/standalone/run_environment_rewrite.py](/home/yyz/projects/Factory_MARL_Project/simulation/standalone/run_environment_rewrite.py)

There are also three external reference lines that influenced the work:

* `test/environment_updated`
* `newtest`
* `newtest2`

The strongest conclusion from all the debugging so far is:

**The current `legacy` environment is still the most integrated line, but it is
not a good long-term base for further environment work.**

The environment should be rebuilt with a clearer structure, not patched
indefinitely.


## 2. Confirmed Facts About the Existing Environment

### 2.1 Static scene source

The static map comes from:

* [simulation/assets/scenes/slam_scene.usda](/home/yyz/projects/Factory_MARL_Project/simulation/assets/scenes/slam_scene.usda)

It contains:

* `World`
* a referenced `PhysicsScene`
* floor geometry
* wall and obstacle geometry
* visual materials


### 2.2 Dynamic actors

In `legacy`, dynamic actors are spawned programmatically:

* two Go2 dogs from `UNITREE_GO2_CFG`
* one humanoid intruder from `HUMANOID_CFG`

See:

* [validate_slam_scene.py](/home/yyz/projects/Factory_MARL_Project/simulation/standalone/validate_slam_scene.py)


### 2.3 Sensors

Current sensor structure in `legacy`:

* dog front cameras:
  * USD camera prims created manually
  * then bound through `CameraCfg(spawn=None, prim_path=...)`
* CCTV cameras:
  * USD camera prims created manually
  * then bound through `CameraCfg(spawn=None, prim_path=...)`
* dog LiDAR:
  * `RayCaster`
* dog IMU:
  * built from articulation/root data for ROS publishing

This means the current environment is a mixed structure:

* static map from USDA
* actors from cfg/articulation spawn
* cameras partly manual USD prims
* other sensors from Isaac Lab objects


### 2.4 Timing

Current default simulation timing:

* physics timestep: `dt = 0.005`
* physics frequency: `200 Hz`
* publish cadence: `publish_every = 4`
* effective state publishing to Core: `50 Hz`

This matches the locomotion training timing:

* locomotion training physics: `200 Hz`
* locomotion policy/control decimation: `50 Hz`

So the main locomotion problem is **not** simply a timing mismatch.


## 3. What Was Confirmed About Perception-Related Environment Data

### 3.1 Semantic segmentation pipeline was initially broken in two ways

It was confirmed and fixed that the previous simulation output had:

* wrong semantic image slicing
* colorized semantic segmentation instead of raw IDs

Those were corrected, and after that:

* semantic map sizes became correct
* semantic unique IDs became non-zero
* `camera info / idToLabels` could be propagated downstream


### 3.2 Camera metadata matters

A direct offline test was done:

* same recorded request
* replace current camera parameters with `newtest` camera parameters
* run current perception offline

Result:

* intruder estimate changed significantly
* confidence increased

This confirmed:

**camera extrinsics/intrinsics are a first-order environment issue.**


### 3.3 The perception core itself was not the main problem

It was confirmed that:

* current repo perception
* `newtest` perception

produce the same intruder result when given the same input.

That means:

**the key environment problem is the simulation-side input quality and camera
configuration, not the perception core implementation by itself.**


## 4. What Was Confirmed About Locomotion

### 4.1 Joint target tracking error is not supposed to be tiny

Using the probe script against the parallel locomotion setup, a baseline run
showed roughly:

* overall mean joint relative error around `0.26`
* overall max joint relative error around `0.59`

So large-looking joint-space gaps between:

* observed relative joint pose
* last applied target relative joint pose

are not automatically evidence of failure.


### 4.2 Velocity tracking in isolation is acceptable

Using the current-project policy in the probe environment, with command:

* `vx = 0.6`
* `vy = 0.0`
* `wz = 0.0`

measured mean velocity tracking was roughly:

* observed `vx ~ 0.61`
* mean absolute `vx` error around `0.02`

This matters because it means:

**the locomotion policy itself is capable of tracking velocity reasonably well
in a clean environment.**


### 4.3 In the current environment, the command is normal but translation is weak

In the real integrated stack, a direct check showed cases like:

* `body_velocity_command ~ [0.43, 0.09, 0.0]`
* locomotion service returned a matching non-trivial velocity
* but actual observed planar speed was only around `0.07`

This matches the user-observed symptom:

* legs are moving
* body translation is weak or stalls

So the main issue is:

**low-level action is being produced, but environment-side contact / physics /
execution conditions are not translating it into forward motion well.**


## 5. Environment Problems That Were Explicitly Identified

### 5.1 Duplicate PhysicsScene in `legacy`

Confirmed:

* `SimulationContext(...)` creates a live simulation physics scene
* `slam_scene.usda` also contains its own `PhysicsScene`

This produced the real warning:

* `Physics scenes stepping is not the same...`

This is not a cosmetic warning. It is an environment design problem.


### 5.2 Floor physics material in the scene did not match training assumptions

The floor in `slam_scene.usda` initially had:

* collision
* visual material

but not an explicit training-aligned rigid-body physics material.

The locomotion training flat environment uses explicit ground material settings
with friction around:

* static friction `1.0`
* dynamic friction `1.0`
* combine mode `multiply`

This mismatch is important for contact behavior.


### 5.3 Robot rigid-body contact material was likely missing training-like friction

The locomotion training environment also randomizes robot rigid-body materials at
startup, effectively fixing the robot-side material to approximately:

* static friction `0.8`
* dynamic friction `0.6`
* restitution `0.0`

That means the training-time contact pair is not just “ground has friction”.

It is closer to:

* ground material defined
* robot material defined

The current `legacy` line did not start from that assumption.


### 5.4 Environment contact conditions are the most likely remaining locomotion issue

After disabling perception output in the control chain, the weak-translation
problem still remained.

So the current evidence says the main issue is not:

* perception output driving the robot wrong

but more likely:

* contact conditions
* collision / floor setup
* scene physics consistency
* environment execution details


## 6. What Was Learned From `test`, `newtest`, and `newtest2`

### 6.1 `environment_updated` / `newtest` structure is cleaner

These reference lines separate concerns more clearly:

* static scene from USDA
* actors from cfg
* CCTV generated in a dedicated helper
* environment packaged as an environment layer
* perception connected by a cleaner contract

This is a better model for a rebuild than the current `legacy` script.


### 6.2 `newtest` and `newtest2` use better camera metadata flow

They rely on:

* semantic tags on spawned actors
* camera-side `idToLabels`
* environment-side camera `info`

This is a better design than reconstructing semantic identity downstream from
partial assumptions.


### 6.3 `newtest2` is especially useful as a perception reference

`newtest2` removed the heavy Isaac Lab runtime dependency from the perception
math path by replacing it with local pure-torch transforms.

That is mainly relevant to perception, but it reinforces a broader point:

**rebuild lines with cleaner boundaries are easier to debug and maintain.**


## 7. Recommended Principles For the Rebuild

The new environment should follow these principles.

### 7.1 Keep the environment layer narrow

The rebuilt environment should own:

* scene loading
* actor spawn
* sensor spawn
* camera metadata
* contact/physics configuration
* simulation publishing

It should **not** absorb:

* perception logic
* decision logic
* planning logic
* locomotion logic


### 7.2 Make the static map authoritative but not self-contained for physics

Use the USDA for geometry and layout, but do not trust it to define the full
runtime physics behavior.

The rebuild should explicitly decide:

* whether the referenced `PhysicsScene` is active
* which scene owns the live physics scene
* what floor physics material is bound
* what robot contact material is bound


### 7.3 Publish full camera metadata as first-class data

The new environment must publish, at least for every relevant camera:

* `rgb`
* `depth` where available
* `semantic_segmentation`
* `info`
* `idToLabels`
* `pos_w`
* `quat_w`
* `intrinsic_matrix`

This should be part of the base environment contract, not an afterthought.


### 7.4 Rebuild contact/material setup explicitly

The rebuild should explicitly define at least:

* floor physics material
* robot rigid-body material
* whether friction combine mode is `multiply`
* restitution

The environment should not rely on “whatever the imported scene happens to do”.


### 7.5 Rebuild using the locomotion training assumptions as a reference

At minimum, the rebuild should align with:

* `dt = 0.005`
* control cadence `50 Hz`
* floor material close to training flat env
* robot rigid-body material close to training startup randomization result

If a deviation is intentional, it should be written down explicitly.


## 8. Recommended Concrete Rebuild Checklist

### Step 1. Start from a clean environment entrypoint

Do not keep extending `validate_slam_scene.py`.

Create a new environment entrypoint whose job is only:

* load scene
* configure runtime physics
* spawn actors
* spawn sensors
* publish outputs


### Step 2. Decide the static scene loading rule

For the new environment:

* reference `slam_scene.usda`
* immediately resolve whether the referenced `PhysicsScene` stays or is disabled

Recommendation:

* use a single runtime-owned physics scene
* disable referenced duplicate scene by design


### Step 3. Define floor and robot contact materials explicitly

Do not inherit these implicitly.

Minimum first pass:

* floor:
  * static friction `1.0`
  * dynamic friction `1.0`
  * restitution `0.0`
  * friction combine `multiply`
* robot:
  * static friction `0.8`
  * dynamic friction `0.6`
  * restitution `0.0`

Then validate locomotion translation before adding more complexity.


### Step 4. Use the cleaner CCTV and camera metadata path

Adopt the `newtest` / `environment_updated` style:

* fixed CCTV helper
* explicit camera poses
* explicit camera metadata capture


### Step 5. Validate with the simplest locomotion-only control loop first

Before reconnecting full perception and MARL:

* run `simulation + navdp + locomotion + core`
* disable perception output usage
* disable MARL output usage

Only once translation looks healthy should the new environment reconnect:

* perception output
* full decision chain


### Step 6. Keep a small probe path for contact/velocity diagnostics

A rebuilt environment should make it easy to inspect:

* root height
* planar speed
* body velocity command
* foot contact state
* last low-level action

This should be part of the debugging path from day one.


## 9. Things Not To Copy Blindly From the Current `legacy` Line

These are exactly the kinds of things that should be reconsidered or rebuilt:

* mixed manual USD camera creation scattered through one large script
* implicit reliance on imported scene physics
* ad hoc contact/material inheritance
* one-file accumulation of environment, actor control, sensor publishing, and
  diagnostics
* environment fixes that are only justified by local patch history


## 10. Things Worth Reusing

These are still useful and should likely be reused in some form:

* `slam_scene.usda` geometry
* current fixed CCTV naming:
  * `cam_nw`, `cam_ne`, `cam_e_upper`, `cam_e_lower`, `cam_se`, `cam_sw`
* current simulation topic contract under `/factory/simulation`
* semantic metadata propagation work:
  * `info`
  * `idToLabels`
  * `suspect_id`
* current intruder route logic as a temporary test tool


## 11. Final Recommendation

The next environment should be rebuilt as a new environment line, not as another
incremental branch of `legacy`.

If we want the shortest reliable path, the new line should be:

1. structurally closer to `environment_updated` / `newtest`
2. explicitly aligned with locomotion training contact assumptions
3. explicit about camera metadata
4. validated first with the simplest possible locomotion-only loop

That is the cleanest way to stop mixing:

* old scene assumptions
* patched runtime behavior
* partial metadata fixes

and finally get to an environment that is easier to trust.
