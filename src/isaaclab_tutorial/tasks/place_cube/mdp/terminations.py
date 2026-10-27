"""Termination terms: confirmed stacking success, a dropped red cube, and a toppled blue cube.

Out-of-workspace cubes and unstable robots use the lift task's ``out_of_bound`` and ``abnormal_robot_state``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

from .geometry import CUBE_REST_HEIGHT
from .progress import stack_progress
from .task import (
    FLAT_ALIGNMENT,
    HARD_STACK_IMPACT_FORCE,
    HOLD_HEIGHT,
    LIFT_CLEARANCE,
    PLACED_HORIZONTAL,
    PLACED_VERTICAL,
    RELEASED_OPENNESS,
    STILL_ANGULAR,
    STILL_LINEAR,
    SUCCESS_HORIZONTAL,
    SUCCESS_VERTICAL,
    blue_on_desk,
    contact_force,
    cube_speeds,
    gripper_openness,
    jaw_contacts,
    red_height,
    stack_state,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def stack_conditions(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
    """The physical predicates behind every milestone, evaluated on the current state."""
    state = stack_state(env)
    fixed, moving = jaw_contacts(env)
    support_height = getattr(env, "_so101_support_height", 0.0)
    red_speed, red_spin = cube_speeds(env, "cube_a")
    blue_speed, blue_spin = cube_speeds(env, "cube_b")
    still = (red_speed < STILL_LINEAR) & (red_spin < STILL_ANGULAR)
    still &= (blue_speed < STILL_LINEAR) & (blue_spin < STILL_ANGULAR)
    vertical = state["offset"][:, 2].abs()
    released = ~(fixed | moving) & (gripper_openness(env) > RELEASED_OPENNESS)
    upright = (state["red_flatness"] > FLAT_ALIGNMENT) & blue_on_desk(env, state)
    centered = (state["horizontal"] <= SUCCESS_HORIZONTAL) & (vertical <= SUCCESS_VERTICAL)
    return {
        "holding": fixed & moving & (red_height(env) > CUBE_REST_HEIGHT + HOLD_HEIGHT + support_height),
        "cleared": state["clearance"] >= LIFT_CLEARANCE,
        "placed": (state["horizontal"] < PLACED_HORIZONTAL) & (vertical <= PLACED_VERTICAL),
        "stacked": centered & upright & released & still,
        "horizontal": state["horizontal"],
    }


class stack_success(ManagerTermBase):
    """Update the task milestones every step and terminate once the red cube sits stacked on the blue cube.

    Milestones a dataset reset starts with (for example a held cube) are restored on reset. The term also records the
    terminal statistics read by :mod:`isaaclab_tutorial.utils.evaluation`.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.progress = stack_progress(env)
        self.max_stack_force = torch.zeros(self.num_envs, device=self.device)
        self.max_support_force = torch.zeros_like(self.max_stack_force)
        self.support_contact_time = torch.zeros_like(self.max_stack_force)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.progress.reset(ids)
        self.max_stack_force[ids] = 0.0
        self.max_support_force[ids] = 0.0
        self.support_contact_time[ids] = 0.0
        if hasattr(self._env, "_so101_reset_grasped"):
            grasped = self._env._so101_reset_grasped[ids]
            self.progress.grasped[ids] = grasped
            self.progress.grasp_count[ids] = torch.where(grasped, self.progress.grasp_steps, 0)
            self.progress.lifted[ids] = self._env._so101_reset_lifted[ids]

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        conditions = stack_conditions(env)
        stack_force = contact_force(env, "cube_stack_contact")
        self.max_stack_force = torch.maximum(self.max_stack_force, stack_force)
        self.progress.unsafe_stack_impact |= stack_force > HARD_STACK_IMPACT_FORCE
        success = self.progress.update(
            conditions["holding"],
            conditions["cleared"],
            conditions["placed"],
            conditions["stacked"],
            env.episode_length_buf,
        )

        log = env.extras.setdefault("log", {})
        log["Metrics/grasp_rate"] = self.progress.grasped.float().mean()
        log["Metrics/lift_rate"] = self.progress.lifted.float().mean()
        log["Metrics/place_rate"] = self.progress.placed.float().mean()
        log["Metrics/success_rate"] = success.float().mean()
        log["Safety/unsafe_stack_impact_rate"] = self.progress.unsafe_stack_impact.float().mean()
        if "fixed_jaw_support" in env.scene.sensors:
            support_force = torch.maximum(
                contact_force(env, "fixed_jaw_support"), contact_force(env, "moving_jaw_support")
            )
            self.max_support_force = torch.maximum(self.max_support_force, support_force)
            self.support_contact_time += (support_force > 1.0).float() * env.step_dt
            log["Safety/peak_jaw_support_force_n"] = self.max_support_force.mean()
            env._so101_terminal_max_support_force = self.max_support_force.clone()
            env._so101_terminal_support_contact_time = self.support_contact_time.clone()

        # Terminal statistics survive the reset that follows, so evaluation can read them.
        env._so101_cube_terminal_progress = torch.stack(
            (self.progress.grasped, self.progress.lifted, self.progress.placed, self.progress.unsafe_stack_impact),
            dim=-1,
        )
        env._so101_cube_terminal_max_stack_force = self.max_stack_force.clone()
        env._so101_cube_terminal_horizontal_offset = conditions["horizontal"].clone()
        env._so101_terminal_time_to_success_s = self.progress.time_to_success.float() * env.step_dt
        return success


def red_dropped(env: ManagerBasedRLEnv, height_above_rest: float = 0.008) -> torch.Tensor:
    """A red cube that was lifted has fallen back to the desk instead of onto the blue cube.

    The 8 mm margin also catches a cube that lands resting on an edge (its center then sits 6 mm higher).
    """
    support_height = getattr(env, "_so101_support_height", 0.0)
    fallen = red_height(env) < CUBE_REST_HEIGHT + support_height + height_above_rest
    return stack_progress(env).lifted & fallen


def blue_toppled(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The blue cube has tipped over, been lifted, or fallen off the desk; nudging it along the desk is allowed."""
    return ~blue_on_desk(env)
