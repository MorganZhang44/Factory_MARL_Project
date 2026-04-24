# Docker and GitHub Management Plan

## Purpose

This document defines how environment sharing and project management should be
handled for the current stage of the Factory MARL project.

The goal is not to force all daily development into containers immediately.

The goal is:

* keep local development fast
* share reproducible environments with teammates
* keep module boundaries aligned with the architecture
* make the project easier to collaborate on through GitHub

---

## Current Working Assumption

At the current stage:

* local development remains the primary workflow
* Docker is used mainly for environment sharing and reproducibility
* GitHub is used for source control, collaboration, and change tracking

This means:

* developers may continue using local Conda environments during active work
* Docker images are updated at stable sharing points, not after every small
  local experiment

---

## Architectural Constraints

This management plan must follow the existing architecture rules.

### 1. Module Boundaries Stay Intact

Each module must still run only in its owned environment.

Current ownership:

* Simulation: `isaaclab51`
* Core and Visualization: `core`
* ROS2 tooling / bringup: `ros2`
* NavDP: `navdp`
* Locomotion: `locomotion`

Docker images should preserve the same ownership model.

### 2. ROS2 Is Not Split Out as a Separate Runtime Service

For the current plan:

* `ros2/` remains as tooling, launch assets, and workspace support
* ROS2 is not treated as an independently deployed application module
* we do not restructure the project around a standalone ROS2 service

### 3. Visualization Stays Inside Core

Visualization remains a Core-owned observability surface.

It is not managed as an independent module image.

---

## Development and Sharing Model

The project uses two different modes on purpose.

### Mode 1: Local Development

Local development remains the main way to work.

Reasons:

* faster iteration
* easier debugging
* easier Isaac Sim usage during active development
* lower friction for scene and integration changes

### Mode 2: Shared Docker Environments

Docker is used to share reproducible environments with teammates.

Reasons:

* easier onboarding
* fewer environment mismatches
* easier baseline reproduction
* more consistent integration testing

This is especially useful when teammates have similar hardware and operating
systems, which is the case for the current team setup.

---

## Docker Update Policy

Docker images do not need to be rebuilt after every small dependency tweak.

That would add overhead without much value.

Instead, Docker should be updated at stable milestones.

### Good Times to Update Docker

Update Docker when:

* a module environment becomes stable enough to share
* a new integration milestone is reached
* a teammate needs to start using that module
* a version is about to be merged into a shared branch

### Not Necessary for Every Small Change

A Docker update is usually not needed when:

* trying a temporary package change
* doing short-lived experiments
* testing multiple environment variants in one day
* changing something that is not ready to share

### Practical Rule

Use this rule:

> local environment changes happen first; Docker is updated when that local
> state becomes worth sharing

---

## Dockerization Priority

Docker should be introduced in stages.

### Stage 1: Standard Service Modules

These modules should be Dockerized first:

* Core
* NavDP
* Locomotion

Reason:

* they behave like standard services
* they are easier to containerize
* they bring immediate sharing value

### Stage 2: Simulation

Simulation should also get a Docker path, but with a more practical target.

The initial goal is not a perfect Isaac Sim container for every use case.

The initial goal is:

* a usable Simulation Docker image for teammates
* focused on reproducibility
* focused on headless or baseline runtime first

---

## Simulation Docker Strategy

Simulation is special and should be treated separately from ordinary Python
service modules.

### Short-Term Goal

Create a first usable Simulation Docker image that can:

* start Isaac Sim / Isaac Lab
* load the current SLAM scene
* run `simulation/standalone/validate_slam_scene.py`
* support the current ROS2 bridge path
* preferably support headless execution first

### Recommended Rollout

#### V1: Usable Headless Baseline

The first Simulation container should prioritize:

* successful startup
* scene loading
* script execution
* baseline reproducibility

This is the best first sharing target for teammates.

#### V2: GUI / Interactive Support

After V1 is stable, improve support for:

* graphical Isaac Sim sessions
* display forwarding
* local interactive debugging

This should be treated as a second step, not the initial requirement.

### Why This Is Reasonable

Because teammates use similar hardware and operating systems, a usable
Simulation Docker image is realistic and worth doing.

But the first version should optimize for "works" rather than "supports every
interactive workflow immediately."

---

## GitHub Management Plan

GitHub should support collaboration without introducing unnecessary process
weight.

### Branch Strategy

Recommended branches:

* `main`: stable and shareable
* `dev`: integration branch
* `feature/*`: module or task branches

### Commit Discipline

Each commit should stay focused.

Good commit grouping:

* one module
* one integration task
* one docs update
* one environment update

Avoid mixing unrelated module changes into a single commit when possible.

### Pull Request Expectations

Each PR should state:

* which module(s) changed
* which runtime chain is affected
* how to run or verify the change
* whether docs were updated

---

## Repository Hygiene

To keep the repo usable over time:

### Keep in Git

Keep these in Git:

* source code
* launch scripts
* environment definitions
* Docker files
* Compose files
* docs
* small configs

### Do Not Keep in Git

Do not keep these in Git:

* large logs
* build outputs
* temporary caches
* heavy model artifacts that are better mounted or downloaded separately

For example:

* large checkpoints
* generated outputs
* local workspace build products

These should be handled through:

* mounted directories
* local setup steps
* release assets or private storage when needed

---

## Recommended Delivery Order

The management work should be done in this order.

### Step 1

Stabilize repository management:

* branch strategy
* commit discipline
* docs updates

### Step 2

Standardize environment descriptions:

* module `environment.yml`
* module README files

### Step 3

Create Docker support for:

* Core
* NavDP
* Locomotion

### Step 4

Add a first usable Simulation Docker baseline:

* headless first
* GUI later

### Step 5

Add Compose support for the shared service modules.

At the current stage, Compose does not need to force local development into
containers. It mainly serves reproducibility and teammate onboarding.

### Step 6

Add minimal GitHub workflow support:

* PR template
* Issue template
* basic CI

---

## Minimal CI Scope

Initial CI should stay lightweight.

Recommended first checks:

* Python syntax checks
* shell syntax checks
* basic docs consistency checks

More advanced integration tests can be added later after container structure and
runtime boundaries become more stable.

---

## Final Position

The management strategy for the current project stage is:

* develop locally
* share stable environments through Docker
* do not split ROS2 into a separate deployed runtime
* treat Simulation Docker as a practical staged effort
* use GitHub to keep collaboration, review, and rollback clean

In short:

> local environments remain the development truth, while Docker becomes the
> reproducible sharing layer.
