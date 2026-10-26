"""SO-101 state and wrist-camera tasks. Add ``presets=sim2real`` to either for robot-transfer randomization."""

import gymnasium as gym

# first we specify the package: the task we are targeting is "place_cube".
#                               In this task we we pick the config folder that contains the robot.
#                               The robot contains the agents and the environment configuration (what we want).
_PACKAGE = "isaaclab_tutorial.tasks.place_cube.config.so101"

# BEGIN lesson-01-registration
# Here we register the task id and we associate it to the task configuration (environment including the robot)
# and the agent configuration
gym.register(
    id="IsaacTutorial-Place-Cube-SO101",  # arbitrarly defined ID
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # entry point pointing to the ManagerBasedRLEnv interface from which
    # our env configuration inherits.
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.env_cfg:SO101CubeEnvCfg",  # our custom env config
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_ppo_cfg:SO101StatePPORunnerCfg",  # our custom PPO config
        "default_agent": "rsl_rl",
    },
)
# END lesson-01-registration

# BEGIN lesson-06-camera-registration
# Task with camera instead of full state observability, policy = distillation
gym.register(
    id="IsaacTutorial-Place-Cube-SO101-Camera",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.camera_env_cfg:SO101CubeCameraEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_distillation_cfg:SO101CameraDistillationRunnerCfg",
        "rsl_rl_ppo_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_ppo_cfg:SO101CameraPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
# END lesson-06-camera-registration
