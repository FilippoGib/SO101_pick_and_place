# Cube-in-Cup Task Migration Plan

## Status

Cube/cup task implementation is wired end-to-end for dataset-backed training
and task-aware evaluation. Reset generation now uses phase-specific SO-101
waypoints and validates every generated row in the actual Newton cube scene
before saving the artifact. The legacy vial task remains unchanged.
Large-scale training and sim-to-real tuning still require empirical policy
validation.

## Confirmed goal

- Replace the vial/rack objective with picking up a 2 cm cube and placing it
  fully inside an upright, open-top regular cup.
- A cube inside a cup is still a failure if the cup is tipped over.
- This is a new task that will be retrained from scratch.
- Preserve the current action and camera/policy interfaces where that remains
  physically meaningful; justify any observation-space changes.
- Prefer an Isaac Lab asset for the cup; otherwise use a small purpose-built
  USD asset or a validated primitive-based cup.

## Implemented shape

Create a cube/cup successor task rather than silently changing the meaning of
the existing vial task:

1. Add a `place_cube` task package and register new state and camera task IDs.
2. Keep the existing vial task and checkpoints reproducible until the new task
   has passed its tests and smoke run.
3. Extract or reuse only robot/control/camera utilities that are genuinely
   object-independent. Avoid a broad speculative refactor.
4. Added a dedicated cube/cup reset schema, generated artifact, reset CLI, and
   reset event wiring.
5. Evaluation now selects vial or cube/cup metrics from the scene entities.

If the intended product is instead to remove the vial task entirely, the same
cube/cup implementation can later replace the old registration and docs in a
small cleanup pass.

## Target files and planned changes

### New or renamed task implementation

- `src/isaaclab_tutorial/tasks/place_cube/__init__.py`
  - Register `IsaacTutorial-Place-Cube-SO101` and
    `IsaacTutorial-Place-Cube-SO101-Camera`.
  - Keep the inspection task shared or leave its registration unchanged.
- `src/isaaclab_tutorial/tasks/place_cube/config/so101/env_cfg.py`
  - Reuse the SO-101 scene, actuators, physics presets, contact setup,
    tabletop/support, reset event structure, and policy/critic wiring.
  - Replace vial and rack scene entities with a 2 cm dynamic cube and an
    upright open-top cup.
  - Configure cube/cup mass, friction, collision geometry, visual materials,
    initial poses, and cup pose randomization.
  - Replace vial-specific event terms and all `SceneEntityCfg("vial")` /
    `SceneEntityCfg("rack")` references.
- `src/isaaclab_tutorial/tasks/place_cube/config/so101/camera_env_cfg.py`
  - Preserve the wrist camera resolution, frame history, proprioception,
    teacher/critic grouping, and camera randomization.
  - Replace vial/rack visual-material handles and camera scene inheritance
    with cube/cup material handles.
- `src/isaaclab_tutorial/tasks/place_cube/config/so101/agents/`
  - Copy/adapt the PPO and distillation entry points only where class/task
    names or experiment identifiers are embedded.
  - Keep action dimensions and network input contracts aligned unless the
    new state features require a measured change.
- `src/isaaclab_tutorial/tasks/place_cube/mdp/task.py`
  - Replace vial/rack geometry constants and helpers with cube/cup frame
    calculations: cube grasp point/extent, cube lowest/highest points, cup
    rim/interior bounds, cup uprightness, cube-in-cup pose, and contact
    queries.
- `src/isaaclab_tutorial/tasks/place_cube/mdp/geometry.py`
  - Replace four-hole/rack geometry with cup-local interior and rim geometry.
  - Add pure tensor predicates for “cube fully inside cup” and “cup upright”.
  - Retain generic quaternion, frame-transform, bounds, and alignment helpers
    where they remain correct.
- `src/isaaclab_tutorial/tasks/place_cube/mdp/observations.py`
  - Rename object-specific terms and update state features from vial/rack
    quantities to cube/cup quantities.
  - Preserve joint, end-effector, previous-action, camera, and progress
    observations.
- `src/isaaclab_tutorial/tasks/place_cube/mdp/rewards.py`
  - Keep approach progress, grasp/lift/placement milestones, success bonus,
    action/joint regularization, and support-force penalty.
  - Replace held insertion shaping with cup-centering/interior-entry shaping.
- `src/isaaclab_tutorial/tasks/place_cube/mdp/terminations.py`
  - Replace vial success with a latched success condition requiring release,
    cube fully inside, cup upright, and both objects stable.
  - Add a tipped-cup failure term and cube-dropped/out-of-bounds handling.
  - Rename task metrics and terminal fields away from rack/hole semantics.
- `src/isaaclab_tutorial/tasks/place_cube/mdp/progress.py`
  - Reuse the milestone state machine shape, changing “inserted” semantics to
    “inside cup” and adding any cup-upright/tipped state needed by termination.
- `src/isaaclab_tutorial/tasks/place_cube/mdp/events.py`
  - Adapt dataset reset loading and object/cup pose writes.
  - Reuse robot calibration, actuator, encoder, and camera randomization.
- `src/isaaclab_tutorial/tasks/place_cube/reset/dataset.py`
  - Use a new format identifier/schema or a clearly separate artifact so a
    vial dataset cannot be loaded accidentally.
  - Store cube and cup poses plus the same robot/phase state needed by resets.
- `src/isaaclab_tutorial/tasks/place_cube/reset/generator.py`
  - Regenerate physics-validated multi-start reset states for cube grasp,
    lift, transport, cup entry, and release.
  - Replace vial/rack waypoint and clearance assumptions with cube/cup
    dimensions and uprightness checks.
- `src/isaaclab_tutorial/tasks/place_cube/reset/curriculum.py`
  - Preserve the phase-balanced curriculum concept, but update phase names
    and canonical-start semantics for the cube/cup motion.

### Assets and asset exports

- `src/isaaclab_tutorial/assets/workshop/cube_2cm.usda`
  - Add a simple, explicitly dimensioned 0.02 m cube with one dynamic body
    and validated collision/visual geometry.
- `src/isaaclab_tutorial/assets/workshop/cup.usda` (or an Isaac Lab asset
  reference if a suitable installed asset is confirmed)
  - Add/use an open-top cup with a real cavity and collision walls; verify
    that the cube cannot succeed through a solid approximation.
  - Define dimensions, bottom, rim, wall thickness, mass, and stable upright
    orientation in one documented asset.
- `src/isaaclab_tutorial/assets/__init__.py`
  - Export cube/cup paths and validate the new dependencies.
- `src/isaaclab_tutorial/assets/reset_poses.pt`
  - Do not overwrite the vial artifact. Add a separate cube/cup reset
    artifact generated by the new reset command.

### Tests

- Update/add config tests for new registrations, scene entities, dimensions,
  observation groups, camera inheritance, and event parameters.
- Replace vial/rack asset tests with cube dimension, cup cavity/collision,
  dependency, and asset-validation tests.
- Replace `test_any_hole.py` with cube-in-cup success/failure cases:
  released vs held, fully inside vs protruding, stable vs moving, upright vs
  tipped cup, and cube/cup relative transforms.
- Adapt geometry, reward, progress, reset-generator, camera-observation, and
  evaluation tests to the new names and predicates.
- Preserve generic robot, projection, action, and utility tests unchanged.
- Add a kit-less config/step smoke test for both state and camera task IDs;
  run a simulator smoke test separately when Isaac Lab is available.

### Utilities, documentation, and experiment metadata

- `src/isaaclab_tutorial/utils/evaluation.py`
  - Generalize or add a cube/cup evaluator with a new result criterion and
    cube/cup terminal fields; retain vial output compatibility if both tasks
    remain registered.
- `README.md`
  - Document the cube/cup task IDs, geometry, success/failure definition,
    retraining commands, reset generation, and camera distillation workflow.
  - Keep the vial section only if the legacy task remains supported.
- `docs/SIM2REAL.md`
  - Update object randomization, physical setup, evaluation, and deployment
    assumptions for the cube/cup task.
- `pyproject.toml`
  - Update project description and add/rename reset-generation entry points
    only if the new task is the primary task; avoid unrelated dependency
    changes.
- `media/` and tracked experiment logs/checkpoints
  - Do not modify old vial artifacts. Add new cube/cup demos or logs only
    after the new task is working.

## Code that should remain the same

- SO-101 USD loading, joint names, relative joint-position action semantics,
  action scales, drive calibration, and robot collider filtering.
- Physics backend presets, contact material registration, solver settings,
  simulation rate, control decimation, and camera rendering pipeline unless
  cube/cup contact stability proves they need targeted tuning.
- Wrist camera resolution/history, proprioceptive observations, corruption
  model, distillation interface, and teacher/student grouping.
- Generic quaternion/frame math, camera projection, reset artifact I/O
  mechanics, phase-balanced sampling, and evaluation callback structure.
- Existing vial task files and artifacts if backward compatibility is kept.

## Reusable code and data concepts

- Reuse the SO-101 scene and action configuration from `env_cfg.py`.
- Reuse contact-material binding and backend-specific contact initialization;
  only object-specific labels/ranges should change.
- Reuse the milestone/reward pattern and `PlacementProgress` state machine,
  with renamed semantics and cup-specific predicates.
- Reuse camera/proprioception classes and distillation runner configuration.
- Reuse reset serialization validation patterns, but not vial poses or vial
  schema identifiers.
- Reuse test helpers for fake Isaac Lab environments and partial-reset
  semantics.

## Cleanest change sequence

1. Confirm or select the cup asset and measure its interior/rim dimensions.
2. Add cube/cup assets and asset validation tests first.
3. Create the new task package and scene/config registrations while preserving
   the vial package.
4. Implement pure cube/cup geometry and unit-test boundary cases.
5. Adapt observations, rewards, progress, and termination around those pure
   predicates.
6. Implement the new reset schema/generator and generate a small validated
   dataset.
7. Add state/camera config tests, then run kit-less tests and a short simulator
   zero-agent smoke test.
8. Run a small training/evaluation smoke experiment; only then update
   documentation/media and decide whether the legacy vial task should be
   removed.

## Decisions requiring justification during implementation

- Whether an installed Isaac Lab cup asset has a usable hollow collision
  volume; a visually cup-shaped but solid collider is not acceptable.
- Whether cube placement should require the cube’s entire axis-aligned or
  oriented bounding box to lie below the cup rim and inside the inner walls.
  The planned default is an oriented-cube containment test in cup coordinates.
- Whether the cup must remain upright throughout placement or only at the
  stable terminal check. The requested behavior requires uprightness at
  success and explicit tipped-cup failure; intermediate tipping can be
  allowed only if the task’s physical objective remains unambiguous.
- Any changed state observation dimensions or terminal metric names, which
  must be reflected in the agent configs, tests, evaluator, and docs together.
