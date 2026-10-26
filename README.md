# SO-101 Isaac Lab Tasks

This repository contains two independent Isaac Lab manipulation tasks for the
SO-101 robot. Both use Newton MJWarp physics, relative joint-position actions,
and optional sim-to-real randomization.

## Setup

Linux x86-64 or Windows 11, Python 3.12, Git LFS, an NVIDIA GPU with a CUDA
13-compatible driver, and `uv` are required.

```bash
git lfs pull
uv sync --locked
uv run pytest -q
```

Add `--extra video` to commands that record video. Add `--extra wandb` and
configure `agent.logger=wandb` for Weights & Biases tracking.

## Cube in cup

The cube task picks up a 2 cm cube and places it fully inside an upright,
slightly conical open-top cup. Success requires release, complete containment,
an upright cup, and stable cube/cup motion. A cube in a tipped cup is not a
success.

### Tasks

| Task | Policy input | Training |
| --- | --- | --- |
| `IsaacTutorial-Place-Cube-SO101` | Full state | PPO |
| `IsaacTutorial-Place-Cube-SO101-Camera` | Wrist RGB, two 64×48 frames, and proprioception | Distillation or PPO |

### Reset dataset

The cube task has its own schema and artifact. Generate it before launching
the environment. Generation validates every generated row in the actual Newton
cube scene before writing the artifact:

```bash
uv run generate-so101-cube-resets \
  --output src/isaaclab_tutorial/assets/cube_reset_poses.pt \
  --poses_per_phase 128 \
  --validation_steps 30 \
  --visualizer none \
  presets=newton_mjwarp
uv run view-so101-cube-resets \
  --dataset src/isaaclab_tutorial/assets/cube_reset_poses.pt
```

The dataset contains robot state, cube pose, cup pose, phase, difficulty, and
latched grasp/lift flags. It is intentionally not compatible with the vial
dataset.

### Smoke test

```bash
uv run isaaclab zero_agent \
  --task IsaacTutorial-Place-Cube-SO101 \
  --num_envs 8 \
  --max_steps 100 \
  --visualizer none \
  presets=newton_mjwarp
```

### Training

```bash
# Full-state cube teacher
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Cube-SO101 \
  --num_envs 4096 \
  --visualizer none \
  presets=newton_mjwarp

# Camera policy distilled from the state teacher
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Cube-SO101-Camera \
  --num_envs 1024 \
  --checkpoint /abs/path/to/cube_state/model.pt \
  --visualizer none \
  presets=newton_mjwarp,newton_renderer
```

For camera PPO from scratch, add
`--agent rsl_rl_ppo_cfg_entry_point` to the camera command.

### Evaluation

The evaluator uses cube/cup-specific output fields for cube tasks:
`inside_cup_rate`, `cube_lost_rate`, `tipped_cup_rate`,
`unsafe_cup_contact_rate`, and cup contact-force metrics.

```bash
SO101_EVALUATION_EPISODES=1024 \
SO101_EVALUATION_OUTPUT=outputs/cube-evaluation.json \
uv run isaaclab play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Cube-SO101 \
  --num_envs 1024 \
  --checkpoint /abs/path/to/cube_state/model.pt \
  --visualizer none \
  --external_callback isaaclab_tutorial.utils.evaluation.install_episode_counter \
  presets=newton_mjwarp
```

The command prints one `SO101_EVAL_RESULT=` JSON line and optionally writes a
per-episode JSON report. Do not pass `--deterministic`; playback already uses
deterministic policy actions.

## Vial in rack (legacy task)

The original task remains available and uses a vial placed upright in any of
four rack openings. Its checkpoints, reset artifact, evaluator fields, and
documentation remain vial-specific.

### Tasks

| Task | Policy input | Training |
| --- | --- | --- |
| `IsaacTutorial-Place-Vial-SO101` | Full state | PPO |
| `IsaacTutorial-Place-Vial-SO101-Camera` | Wrist RGB, two 64×48 frames, and proprioception | Distillation or PPO |

### Reset dataset

```bash
uv run generate-so101-resets \
  --output outputs/vial_reset_poses.pt \
  --visualizer none \
  presets=newton_mjwarp
uv run view-so101-resets \
  --dataset outputs/vial_reset_poses.pt \
  --visualizer newton \
  presets=newton_mjwarp
```

### Training and evaluation

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 4096 \
  --visualizer none \
  presets=newton_mjwarp

SO101_EVALUATION_EPISODES=1024 \
uv run isaaclab play --rl_library rsl_rl \
  --task IsaacTutorial-Place-Vial-SO101 \
  --num_envs 1024 \
  --checkpoint /abs/path/to/vial_state/model.pt \
  --visualizer none \
  --external_callback isaaclab_tutorial.utils.evaluation.install_episode_counter \
  presets=newton_mjwarp
```

Vial evaluation retains its original rack/hole metrics, including
`insertion_rate`, `unsafe_rack_contact_rate`, and `successes_by_hole`.

## Shared design

- Actions are relative joint-position targets at 30 Hz; physics runs at 120 Hz.
- `presets=sim2real` enables robot-transfer randomization.
- Camera tasks use the same wrist-camera and proprioception interface as their
  state teacher.
- PPO and camera distillation checkpoints are written under `logs/rsl_rl/`.

The implementation is organized by task:

```text
src/isaaclab_tutorial/
  assets/workshop/                         USD assets
  tasks/place_vial/                        legacy vial/rack task
  tasks/place_cube/                        cube/cup task
  utils/evaluation.py                      task-aware exact episode counter
tests/                                     unit and configuration tests
```
