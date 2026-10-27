"""Observation terms: task-specific state features for the stacking teacher."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .progress import stack_progress
from .task import cube_speeds, jaw_contacts, stack_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_target(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint-position targets currently sent to the drives [rad]."""
    return env.scene[asset_cfg.name].data.joint_pos_target.torch[:, asset_cfg.joint_ids]


def encoder_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint positions as the robot's encoders report them, including the sampled calibration bias [rad]."""
    return env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids] + getattr(
        env, "_so101_encoder_bias", 0.0
    )


def encoder_joint_target(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint targets in the same (biased) encoder coordinates as :func:`encoder_joint_pos` [rad]."""
    return joint_target(env, asset_cfg) + getattr(env, "_so101_encoder_bias", 0.0)


def stack_features(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Red offset from the stacked pose (xyz), its horizontal norm, clearance over the blue top, both cubes'
    flatness, and the red cube's speed."""
    state = stack_state(env)
    speed, _ = cube_speeds(env, "cube_a")
    return torch.cat(
        (
            state["offset"],
            torch.stack(
                (state["horizontal"], state["clearance"], state["red_flatness"], state["blue_flatness"], speed),
                dim=-1,
            ),
        ),
        dim=-1,
    )


def progress_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Latched grasp, lift and placement milestones, which make the once-per-episode rewards Markov."""
    progress = stack_progress(env)
    return torch.stack((progress.grasped, progress.lifted, progress.placed), dim=-1).float()


def jaw_contact_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Privileged fixed-jaw, moving-jaw and bilateral contact."""
    fixed, moving = jaw_contacts(env)
    return torch.stack((fixed, moving, fixed & moving), dim=-1).float()
