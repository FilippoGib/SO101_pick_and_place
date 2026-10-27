"""SO-101 two-cube scene. Add ``presets=sim2real`` for the tabletop base height, gray mat and support sensing."""

import gymnasium as gym

_PACKAGE = "isaaclab_tutorial.tasks.place_cube.config.so101"

# Actions, observations and terminations without rewards: runs zero and random agents, not training yet.
gym.register(
    id="IsaacTutorial-Place-Cube-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{_PACKAGE}.env_cfg:SO101CubeEnvCfg"},
)

# Scene-only task with no actions, used to inspect the cubes before a task is defined.
gym.register(
    id="IsaacTutorial-Inspect-Cube-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{_PACKAGE}.inspection_env_cfg:SO101CubeInspectionEnvCfg"},
)
