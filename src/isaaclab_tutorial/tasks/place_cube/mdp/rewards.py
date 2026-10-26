"""Dense shaping and milestone rewards for cube placement."""

from collections.abc import Sequence

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg

from .progress import placement_progress
from .task import cube_grasp_point_w, cube_on_cylinder, grasp_center_w


def _finite_reward(value: torch.Tensor) -> torch.Tensor:
    return value.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)


def action_rate_l2(env):
    from isaaclab.envs import mdp

    return _finite_reward(mdp.action_rate_l2(env))


def joint_velocity_l2(env, asset_cfg=None):
    from isaaclab.envs import mdp

    if asset_cfg is None:
        return _finite_reward(mdp.joint_vel_l2(env))
    return _finite_reward(mdp.joint_vel_l2(env, asset_cfg))


def joint_position_limits(env, asset_cfg=None):
    from isaaclab.envs import mdp

    if asset_cfg is None:
        return _finite_reward(mdp.joint_pos_limits(env))
    return _finite_reward(mdp.joint_pos_limits(env, asset_cfg))


class approach_progress(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self.previous_distance = torch.full((self.num_envs,), float("nan"), device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        self.previous_distance[slice(None) if env_ids is None else env_ids] = float("nan")

    def __call__(self, env, scale: float = 0.01):
        distance = torch.linalg.vector_norm(grasp_center_w(env) - cube_grasp_point_w(env), dim=-1)
        progress = ((self.previous_distance - distance) / scale).clamp(-1.0, 1.0)
        self.previous_distance = distance
        return torch.where(placement_progress(env).grasped, 0.0, progress.nan_to_num(0.0))


def held_goal(env, std: float = 0.03):
    progress = placement_progress(env)
    error = cube_on_cylinder(env)
    shaping = torch.exp((-error / std).clamp(min=-80.0, max=0.0))
    return torch.where(progress.grasped & ~progress.inside & torch.isfinite(error), shaping, 0.0)


class milestones(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.paid = torch.zeros((self.num_envs, 3), dtype=torch.bool, device=self.device)
        self.started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.paid[ids] = False
        self.started[ids] = False

    def __call__(self, env):
        progress = placement_progress(env)
        reached = torch.stack((progress.grasped, progress.lifted, progress.inside), dim=-1)
        reward = ((reached & ~self.paid).float() * reached.new_tensor((1.0, 2.0, 4.0))).sum(-1)
        reward = torch.where(self.started, reward, 0.0)
        self.paid = reached.clone()
        self.started[:] = True
        return reward


def success(env):
    return placement_progress(env).success.float()


def jaw_support_force(env, free_force: float = 1.0, force_span: float = 5.0):
    from .task import contact_force

    force = torch.maximum(contact_force(env, "fixed_jaw_support"), contact_force(env, "moving_jaw_support"))
    return _finite_reward(((force - free_force) / force_span).clamp(0.0, 1.0))
