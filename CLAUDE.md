# CLAUDE.md — project notes

## What this is

An Isaac Lab 3.0 tutorial package (`isaaclab_tutorial`): train an **SO-101** arm (WowRobo follower, 5 arm joints + gripper)
to pick a vial lying on a desk and place it upright in **any of the 4 holes** of a rack.

Pipeline: **state teacher (PPO)** → **wrist-camera student (DAgger-style distillation)** → LEAPP export → real robot via
LeRobot. Physics is **Newton MJWarp**; camera rendering is the **Newton renderer**. PhysX/OVRTX are optional sim-to-sim
backends and are not tuned.

The repo came from NVIDIA's "Getting Started With Isaac Lab → Train a Robot Arm" course. The `# BEGIN lesson-XX-...` /
`# END lesson-XX-...` markers show where each lesson's code goes. All lessons are already filled in on `main`, so the code
is the complete reference. The README's first paragraph still describes the exercise scaffold, which is out of date.
The user has added their own explanatory comments in some lesson regions (for example in `config/so101/__init__.py`).
Keep the markers and those comments.

## Branches / ongoing work

- `main`: one commit, the completed vial task. `dev/cube_active` (the current branch) starts out identical to `main`.
- `dev/cube_abandoned`: an earlier attempt at a **cube task**. It added a separate `tasks/place_cube/` package plus
  `cube_2cm.usda`, `cup.usda`, `cube_tray.usda`, `cylinder_3cm.usda`, `cube_reset_poses.pt`, `tests/test_cube_task.py`,
  and a planning doc `agent.md`. Its history:
  "cube in cup — training instability, maybe geometry too complex?" → "cube on cylinder checkpoint" → "model not learning".
  To see it: `git show dev/cube_abandoned:agent.md` and `git diff main dev/cube_abandoned`.
- `dev/cube_active` now has `tasks/place_cube/`: red-on-blue stacking, ready for teacher training. See "place_cube" below.
- The branch name suggests a fresh cube-manipulation attempt will happen here. Learn from the abandoned branch: keep the
  first target geometry simple, and validate the reset dataset and rewards before scaling up training.

## Commands (always through `uv`, Python 3.12, CUDA 13)

```bash
uv sync --locked                       # extras: --extra video | wandb | leapp | ovphysx | ovrtx
uv run pytest -q                       # markers: unit, integration, smoke, kitless
uv run ruff check . && uv run ruff format .   # line length 120; pre-commit also runs codespell
# smoke test that the task builds and steps
uv run isaaclab zero_agent --task IsaacTutorial-Place-Vial-SO101 --num_envs 8 --max_steps 100 --visualizer none presets=newton_mjwarp
# train the teacher (800 iters) / distill the student (1600 iters, --checkpoint = teacher .pt)
uv run isaaclab train --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101 --num_envs 4096 --visualizer none presets=newton_mjwarp
uv run isaaclab train --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101-Camera --num_envs 1024 --checkpoint <abs teacher.pt> --visualizer none presets=newton_mjwarp,newton_renderer
# exact evaluation: 1 episode per env from the home pose, prints a SO101_EVAL_RESULT JSON line
uv run isaaclab play ... --external_callback isaaclab_tutorial.utils.evaluation.install_episode_counter presets=...
# reset dataset tools
uv run generate-so101-resets --output outputs/reset_poses.pt --visualizer none presets=newton_mjwarp
uv run view-so101-resets --dataset outputs/reset_poses.pt --visualizer newton presets=newton_mjwarp
```

- Never pass `--deterministic` to play. It enables Warp deterministic mode, which overflows a collision-kernel buffer.
- Presets are Hydra-style overrides: `presets=newton_mjwarp[,newton_renderer][,sim2real]`. In code,
  `preset(default=..., sim2real=..., physx=...)` from `isaaclab_tasks.utils` selects per-preset values, and
  `PresetCfg` subclasses do the same for whole configs (see `PhysicsCfg`).
- `resume_training_distillation.txt` holds the user's command for resuming distillation (`--resume --run_name distillation`).

## Layout

```
src/isaaclab_tutorial/
  tasks/__init__.py                  import_packages(); exposed through the entry point "isaaclab.tasks" in pyproject
  tasks/place_vial/
    config/so101/__init__.py         gym.register for the 3 task IDs (below)
    config/so101/env_cfg.py          (~840 lines) the core: constants, contact material, SO-101 spawn,
                                     SO101SceneCfg, ActionsCfg, obs groups, EventsCfg/Sim2RealEventsCfg,
                                     RewardsCfg, TerminationsCfg, PhysicsCfg, SO101VialEnvCfg (+ play_mode)
    config/so101/camera_env_cfg.py   SO101CameraSceneCfg (wrist camera), WristImageCfg, ProprioceptionCfg,
                                     CameraSim2RealEventsCfg, SO101VialCameraEnvCfg (inherits the state env)
    config/so101/inspection_env_cfg.py + lesson_scaffold.py   inspection task without actions (lessons 02–04)
    config/so101/agents/             rsl_rl_ppo_cfg.py (state + camera PPO), rsl_rl_distillation_cfg.py
    mdp/                             __init__.py uses lazy_export(); __init__.pyi re-exports Isaac Lab core and
                                     lift-task terms plus local ones. **Add new terms to __init__.pyi.**
      task.py          vial/rack/gripper quantities and constants (fingertip offsets, rim height, insertion radius…)
      geometry.py      pure tensor math (rack hole centers, quaternions in XYZW, bounds) — unit-testable without the sim
      progress.py      PlacementProgress: per-env latched milestones grasped/lifted/inserted, partial-reset safe
      rewards.py       approach_progress, held_goal, milestones, success, jaw_support_force
      terminations.py  placement_success (released, upright, still for 10 steps), vial_dropped
      observations.py  joint_target, encoder terms, vial_in_rack_frame, placement_features, progress_flags, wrist_rgb
      events.py        ResetFromDataset, phase-balanced sampling, viscous friction / encoder bias / camera-mount randomization
    reset/
      dataset.py       .pt format "so101-vial-reset-dataset" schema v2, with validation and a content digest
      generator.py     (~2250 lines) scripted waypoint generator + viewer, physics-validates every row
      curriculum.py    8 phases: home, pregrasp, grasp, lift, reorient, transport, insert, release;
                       CANONICAL_START (home only) vs ALL_PHASES
  assets/__init__.py                 paths: VIAL_USD, VIAL_LABELED_USD, RACK_USD, DESK_USD, CUBE_USD, RESET_DATASET (+ validate_assets)
  assets/workshop/*.usda             desk, rack, rack_simple, vial, vial_labeled, vial_opaque, cube_3cm
  tasks/place_cube/                  red-on-blue stacking teacher: mdp/, reset/, agents/ (see below)
  assets/reset_poses.pt              tracked through LFS; the only .pt file that is not gitignored
  utils/evaluation.py                exact episode counter for play (env SO101_EVALUATION_EPISODES / _OUTPUT)
  utils/export_leapp.py              exports the camera actor as a LEAPP bundle
  utils/deploy.py, calibrate.py, home_so101.py, inspect_so101.py
                                     real-robot scripts with inline uv script metadata (CPU torch, LeRobot 0.6.1);
                                     run with `uv run --script`, Isaac Lab not required
config/so101_joint_map.example.json, so101_start_pose.json   real-robot joint mapping and home pose
docs/SIM2REAL.md                     sim2real training recipe, physical setup, deployment steps
tests/                               mostly kit-less unit tests (geometry, rewards, progress, reset dataset, config…)
```

Untracked, gitignored local outputs: `logs/rsl_rl/{so101_vial_state,so101_vial_camera_distillation}/…`, `videos/`,
`outputs/`, `evaluation-distillation/student.log`. Tracked: `evaluation/teacher_exp1.txt`.

## Tasks

| ID | Env cfg | Agent |
| --- | --- | --- |
| `IsaacTutorial-Place-Vial-SO101` | `SO101VialEnvCfg` | `SO101StatePPORunnerCfg` (exp `so101_vial_state`) |
| `IsaacTutorial-Place-Vial-SO101-Camera` | `SO101VialCameraEnvCfg` | default `SO101CameraDistillationRunnerCfg`; `--agent rsl_rl_ppo_cfg_entry_point` → `SO101CameraPPORunnerCfg` |
| `IsaacTutorial-Inspect-SO101` | `SO101InspectionEnvCfg` | none (no actions) |
| `IsaacTutorial-Inspect-Cube-SO101` | `place_cube…SO101CubeInspectionEnvCfg` | none (no actions) |
| `IsaacTutorial-Place-Cube-SO101` | `place_cube…SO101CubeEnvCfg` | `SO101CubeStatePPORunnerCfg` (exp `so101_cube_state`) |

## place_cube: red-on-blue stacking state teacher (ready to train)

**Task:** pick up the red cube (`cube_a`) and place it on the blue cube (`cube_b`).
- Success, held for **10 consecutive steps**:
  - red center within **10 mm** horizontally of blue's center, 30 ± 4 mm above it
  - red flat (a face within 15° of up)
  - blue flat on the desk
  - both cubes calm (< 0.05 m/s, < 0.5 rad/s)
  - jaws released (no contact, gripper > 20% open)
  - any yaw is accepted
- Failures: `red_dropped` (lifted, then back at desk height), `blue_toppled` (tilt > 25° or off desk height; nudges
  are fine), either cube out of bounds. All of them pay `cube_lost` = −50.

**Structure**
- `env_cfg.py` is a **copy** of the vial task's robot, desk, contact model, mat and `PhysicsCfg`; the user chose this.
  Fixes to shared infrastructure must be made in both copies. Both modules register a Newton MODEL_INIT contact
  callback under different names; when both are imported, both run and write identical values.
- **Scene:** robot, desk, `cube_a` red at (0.231, −0.017), `cube_b` blue at the rack footprint center (0.2102, 0.1101).
  Cubes start at z 0.0505 and settle to 0.050.
  - Sensors: `fixed_jaw_contact` (unfiltered), `moving_jaw_contact` (filtered to **CubeA only**), `cube_stack_contact`
    (CubeA vs CubeB, a metric only).
  - `sim2real` adds the vial scene variants: base on the 35 mm tabletop, shoulder/elbow fits, gray mat, jaw-support
    sensors.
- **Asset:** `assets/workshop/cube_3cm.usda` (`CUBE_USD`) is a 30 mm, 20 g rigid body with a `Color` variant (red/blue).
  Colors exist only inside the variants. Kitless Isaac Lab ignores `visual_material` overrides.
- **`place_cube/mdp/`:** `lazy_export` + `__init__.pyi`, on the `import_packages` skip list.
  - `geometry.py`: pure math. `CUBE_SIZE`, `DESK_TOP_Z`, `CUBE_REST_HEIGHT`, flatness, stack offset.
  - `task.py`: all thresholds plus `stack_state`.
  - `progress.py`: `StackProgress` (grasped/lifted/placed/success). Lifted, placed and success **require a grasp**.
  - `observations.py`, `rewards.py`, `terminations.py`, `events.py`.
  - Tests import the submodules by full dotted path, because the lazy `mdp` package also re-exports Isaac Lab's
    `terminations`/`rewards` modules by name.
- **Observations:**
  - `policy` (**74** values): joint pos/vel/target, last action, gripper `body_state_b`, red and blue `body_state_b`,
    `stack_features` (8), progress flags (3).
  - `critic`: policy + jaw contact flags, **77** values.
- **Rewards** (vial weights): approach_progress 1, held_goal 0.1, milestones 10 (grasp : lift : place = 1 : 2 : 4),
  success 200, cube_lost −50, action_rate −0.002, joint_vel −0.0002, joint_limits −1, plus jaw_support −8 in sim2real.
- **Events:**
  - Nominal: startup red/blue friction 0.7–1.3, red mass 15–25 g, `ResetFromDataset` with all phases.
  - `Sim2RealEventsCfg`: per-reset materials, mass, gains, joint friction, viscous friction and encoder bias; home
    starts only, with ±1 cm / ±10° red and ±5 mm / ±5° blue noise; mat height.
- **`play_mode()`:** sequential home starts. Nominal mode pins cube friction to 0.7 and red mass to 20 g.
- **Agent:** `agents/rsl_rl_ppo_cfg.py:SO101CubeStatePPORunnerCfg` is identical to the vial teacher except for
  `experiment_name=so101_cube_state` and `max_iterations=1500`.

**Reset dataset** (`assets/cube_reset_poses.pt`, `CUBE_RESET_DATASET`)
- Format `so101-cube-stack-reset-dataset` v1. 7 phases × 128 rows: home, approach, grasp, lift, transport, place,
  release. Committed as a plain blob (`.gitignore`/`.gitattributes` exceptions).
- `place_cube/reset/generator.py` (`uv run generate-so101-cube-resets`, about 20 s on GPU; viewer
  `view-so101-cube-resets [--phase NAME]`): every row is a physically simulated, validated rollout. Seed banks chain
  phases. Grasp rows are restored and re-verified. Flags are measured, not inferred from the phase.
- **Hard-won constraints** (don't regress these):
  1. **IK targets are in the environment frame.** The IK model is the env-0 robot moved to the origin.
  2. **A straight-down grasp is unreachable** here: `wrist_flex` exceeds its ±1.62 rad limit. The gripper is pitched
     **15°** about the lateral axis, *in the radial plane through the base*, the only tilt a 5-DoF arm can make at any
     roll. Carries rotate the target with the pan angle so it stays in that plane.
  3. The drives stop about 1° short (system-identified Coulomb friction acts as a deadband), so the TCP misses by about
     6 mm. Carries use Cartesian feedback, and the proof lift is 20 mm.
  4. Carry height is 0.095 m (red center): above that, `wrist_flex` runs out of margin.
- Acceptance per 1024 batch: approach 99%, grasp 78%, lift 60%, transport 53%, place 86%, release 100%. Rows after
  the lift lose the farthest red layouts (x > 0.25) to the wrist limit; home rows cover the full range.

**Validated**
- 141 tests pass.
- Replay: every row restores once with matching flags, nothing terminates on step 1, and no restored milestone is paid.
  Holding the recorded targets for 1 s keeps 100% of held rows held. Release rows succeed on their own; that's
  intended, as in the vial task.
- random_agent with both presets; PPO smoke run (512 × 30 iterations); `play` with the episode counter
  (`success_criterion=red_on_blue_v1`).

**Commands**
- Train: `uv run isaaclab train --rl_library rsl_rl --task IsaacTutorial-Place-Cube-SO101 --num_envs 4096 --visualizer none presets=newton_mjwarp`
- Evaluate: the vial `play` command with this task ID (prints `SO101_EVAL_RESULT` with grasp/lift/place/success,
  red_dropped, blue_toppled and stack-force fields).
- Inspect: `uv run isaaclab zero_agent --task IsaacTutorial-Inspect-Cube-SO101 --num_envs 1 --max_steps 600 --visualizer newton presets=newton_mjwarp`

## Task design facts (check these before changing anything)

- **Timing:** sim dt 1/120 s, decimation 4 → 30 Hz policy. Episodes last 20 s (30 s with sim2real).
- **Actions:** `RelativeJointPositionActionCfg`, where the target is the *measured* position plus scale × action. Scale is
  0.033 rad for the arm (sim2real: shoulder_lift 0.12, elbow_flex 0.07) and 0.02 for the gripper. Actions are clipped to
  [-1, 1]. 6-D action.
- **Joints:** `JOINTS = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`. The robot base is
  yawed +90° about Z, and quaternions in configs are **XYZW**.
- **Scene entities:** `robot`, `vial` (RigidObject, starts lying horizontal at (0.231,-0.017,0.06)), `rack` (at
  (0.18,0.08,0.04)), `desk`, optional `support` (gray mat, sim2real only), and contact sensors
  `fixed_jaw_contact`, `moving_jaw_contact`, `vial_rack_contact`, plus `*_jaw_support` in sim2real.
  Non-gripper robot colliders are removed at spawn time (`_remove_non_gripper_colliders`).
- **Observations:** the `policy` group holds joint pos/vel/target, last action, EE pose, vial pose, vial-in-rack,
  placement features and progress flags. `critic` is `policy` plus jaw contacts. The camera task has the groups
  `wrist_rgb` (2-frame history; rendered at 80×60, network input 64×48), `proprioception`, `teacher_state` and `critic`.
- **Rewards:** approach_progress 1.0, held_goal 0.1, milestones 10 (grasp/lift/insert, each paid once), success 200,
  vial_lost -50, action_rate -0.002, joint_vel -2e-4, joint_limits -1, and jaw_support_force -8 in sim2real.
- **Resets:** `ResetFromDataset` samples from `reset_poses.pt` across all 8 phases during training. Play/eval and sim2real
  start from home only (`CANONICAL_START`). `play_mode()` caps the env count and pins the vial's friction and mass.
- **Networks:** the teacher MLP is [256,256,128] ELU with log-std clamped to [0.05,0.3]. The student CNN is [16,32,32]
  with an MLP [256,128]. The distillation teacher config **must mirror** the state actor, or the checkpoint won't load.
- **Physics:** MJWarp tuning (njmax 300, nconmax 200, 50 iterations, implicitfast, 2 substeps) and custom contact
  stiffness/damping/solref live in `env_cfg.py`. The vial mass is set explicitly because Newton can't infer it from the mesh.

## Baseline results (local runs)

- State teacher `logs/rsl_rl/so101_vial_state/2026-10-23_16-11-59_state-teacher/model_799.pt`: 99.8% success over
  1024 episodes, mean 7.7 s to success. Every success went into hole 0 (`successes_by_hole: [1022,0,0,0]`).
- Distilled student: 99.8% success, lift rate 0.92.

## Conventions when editing

- Use `@configclass` configs. Pick variants with `preset(...)` rather than `if` branches. Refer to scene entities with
  `SceneEntityCfg("name")`.
- Keep pure math in `geometry.py`, testable without the sim. Tests are mostly kit-less and use fake envs.
- Docstrings are short and comments are sparse and explain *why*. Ruff isort sections are configured
  (isaaclab → isaaclab_rl → isaaclab_tasks → isaaclab_assets → first-party).
- Changing an observation's dimension or meaning ripples into the agent configs, the distillation teacher mirror, the
  evaluation fields, tests, `export_leapp.py`/`deploy.py` (for camera/proprio inputs) and the docs. Update them together.
- The reset dataset is tied to the scene geometry. If objects or poses change, regenerate it, or add a new schema/format
  ID so the wrong dataset can't load.
- Don't overwrite tracked assets (`reset_poses.pt`, USDs) or the user's training logs. Add new files next to them instead.
