"""Cube-in-cup success and failure conditions."""

from collections.abc import Sequence

import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

from .progress import placement_progress
from .task import (
    CUBE_REST_HEIGHT,
    contact_force,
    cube_contained,
    cube_height,
    cube_upright,
    gripper_openness,
    jaw_contacts,
)


class placement_success(ManagerTermBase):
    def __init__(self, cfg: TerminationTermCfg, env):
        super().__init__(cfg, env)
        self.progress = placement_progress(env)
        self.max_cup_force = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = slice(None) if env_ids is None else env_ids
        self.progress.reset(ids)
        self.max_cup_force[ids] = 0.0
        if hasattr(self._env, "_so101_reset_grasped"):
            grasped = self._env._so101_reset_grasped[ids]
            self.progress.grasped[ids] = grasped
            self.progress.grasp_count[ids] = torch.where(grasped, self.progress.grasp_steps, 0)
            self.progress.lifted[ids] = self._env._so101_reset_lifted[ids]

    def __call__(self, env):
        fixed, moving = jaw_contacts(env)
        released = ~(fixed | moving) & (gripper_openness(env) > 0.2)
        contained = cube_contained(env)
        upright = cube_upright(env).bool()
        cube = env.scene["cube"]
        cup = env.scene["cup"]
        still = (torch.linalg.vector_norm(cube.data.root_lin_vel_w.torch, dim=-1) < 0.06) & (
            torch.linalg.vector_norm(cube.data.root_ang_vel_w.torch, dim=-1) < 0.8
        )
        cup_still = (torch.linalg.vector_norm(cup.data.root_lin_vel_w.torch, dim=-1) < 0.06) & (
            torch.linalg.vector_norm(cup.data.root_ang_vel_w.torch, dim=-1) < 0.8
        )
        seated = contained & upright & released & still & cup_still
        impact = contact_force(env, "cube_cup_contact")
        self.max_cup_force = torch.maximum(self.max_cup_force, impact)
        success = self.progress.update(
            fixed & moving & (cube_height(env) > CUBE_REST_HEIGHT + 0.006),
            cube_height(env) > 0.08,
            contained,
            seated,
            upright,
            env.episode_length_buf,
        )
        log = env.extras.setdefault("log", {})
        log["Metrics/grasp_rate"] = self.progress.grasped.float().mean()
        log["Metrics/lift_rate"] = self.progress.lifted.float().mean()
        log["Metrics/inside_cup_rate"] = self.progress.inside.float().mean()
        log["Metrics/success_rate"] = success.float().mean()
        self.progress.tipped_cup |= ~upright
        env._so101_terminal_progress = torch.stack(
            (self.progress.grasped, self.progress.lifted, self.progress.inside, self.progress.tipped_cup), dim=-1
        )
        env._so101_terminal_max_cup_force = self.max_cup_force.clone()
        return success


def cube_dropped(env, drop_clearance: float = 0.01):
    """Terminate a lifted episode when the cube returns to the tabletop."""
    returned_to_table = cube_height(env) < CUBE_REST_HEIGHT + drop_clearance
    return placement_progress(env).lifted & ~cube_contained(env) & returned_to_table


def cup_tipped(env):
    return placement_progress(env).inside & ~cube_upright(env).bool()
