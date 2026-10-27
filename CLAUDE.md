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
- `dev/cube_active` now has `tasks/place_cube/`, which has reached the lesson-04 stage (runs random/zero agents, no rewards yet); see "place_cube" below.
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
  tasks/place_cube/                  two-cube scene + lesson-04 env, no rewards yet (see below)
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
| `IsaacTutorial-Place-Cube-SO101` | `place_cube…SO101CubeEnvCfg` | none yet (random_agent / zero_agent only) |

## place_cube (lesson-04 stage: no rewards, not trainable yet)

- `tasks/place_cube/config/so101/env_cfg.py` is a **copy** of the vial task's robot, desk, contact model, mat and
  `PhysicsCfg`, so it is independent of `place_vial` (the user chose this). Fixes to shared infrastructure have to be
  made in both copies. Both modules register a Newton MODEL_INIT contact callback under different names; when both
  are imported, both run and write identical values.
- `SO101CubeSceneCfg` contains robot, desk, `cube_a` (red) at the vial position (0.231, -0.017) and `cube_b` (blue) at
  the rack footprint's center (0.2102, 0.1101). Both cubes start at z 0.0505 and settle to 0.050. The scene also has
  `fixed_jaw_contact`, plus `moving_jaw_contact` filtered against both cubes. `sim2real` adds the same scene variants as
  the vial task (base on the 35 mm tabletop, shoulder/elbow fits, gray mat, jaw-support sensors), but **no** per-episode
  randomization events yet.
- `assets/workshop/cube_3cm.usda` (`CUBE_USD`) is a 30 mm, 20 g rigid body with color variants `Color` = red/blue.
  Colors are authored only inside the variants: local opinions would override them. Isaac Lab ignores the
  `visual_material` override in kitless mode, so the cubes are colored through variants instead.
- `SO101CubeEnvCfg` (in `env_cfg.py`) mirrors `SO101VialEnvCfg` at the lesson-04 stage.
  - `ActionsCfg` is identical to the vial task's, including the sim2real scales.
  - The `policy` observations are joint pos/vel (6+6), last action (6) and `body_state_b` for `cube_a`/`cube_b`
    (13 each), giving **44** values. There is no `critic` group yet.
  - `EventsCfg` only calls `reset_scene_to_default`.
  - Terminations are `cube_a/b_out_of_bounds` (with the vial workspace bounds; falling off the desk trips the z bound),
    `unstable_robot` and `time_out`.
  - `rewards = None` and there is no `play_mode()`. Episodes last 20 s (30 s with sim2real); `record_video` writes to
    `videos/simple_agents_cube`.
- `place_cube/mdp/` follows the vial convention: `lazy_export()` + `__init__.pyi`, which re-exports
  `isaaclab.envs.mdp.*` + lift `abnormal_robot_state`, `body_state_b`, `out_of_bound`. It is on the `import_packages`
  skip list in `tasks/__init__.py`. **Put new cube terms here and list them in `__init__.pyi`.** There is no `reset/`
  or agent config yet. The inspection env still imports `isaaclab.envs.mdp` directly.
- Validated: random_agent on 64 envs × 600 steps, with and without sim2real. Moving a cube under the desk fires only that
  env's `cube_a_out_of_bounds`, and the reset restores the default pose. Uniform random relative actions only make the
  arm jitter near its start pose, so they never knock a cube off the desk.
- Inspect: `uv run isaaclab zero_agent --task IsaacTutorial-Inspect-Cube-SO101 --num_envs 1 --max_steps 600 --visualizer newton presets=newton_mjwarp`
- Random agent: `uv run isaaclab random_agent --task IsaacTutorial-Place-Cube-SO101 --num_envs 16 --visualizer newton presets=newton_mjwarp`

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
