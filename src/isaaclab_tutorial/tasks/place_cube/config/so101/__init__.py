"""SO-101 red-on-blue cube stacking and its inspection scene. Add ``presets=sim2real`` for robot transfer."""

import gymnasium as gym

_PACKAGE = "isaaclab_tutorial.tasks.place_cube.config.so101"

# Red-on-blue stacking from full state: the PPO state teacher.
gym.register(
    id="IsaacTutorial-Place-Cube-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.env_cfg:SO101CubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_ppo_cfg:SO101CubeStatePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# BEGIN lesson-06-camera-registration
# Wrist camera and proprioception instead of full state: distilled from the state teacher, or PPO from scratch.
gym.register(
    id="IsaacTutorial-Place-Cube-SO101-Camera",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.camera_env_cfg:SO101CubeCameraEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_distillation_cfg:SO101CubeCameraDistillationRunnerCfg",
        "rsl_rl_ppo_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_ppo_cfg:SO101CubeCameraPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
# END lesson-06-camera-registration

# Scene-only task with no actions, used to inspect the cubes before a task is defined.
gym.register(
    id="IsaacTutorial-Inspect-Cube-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{_PACKAGE}.inspection_env_cfg:SO101CubeInspectionEnvCfg"},
)
