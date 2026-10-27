"""Cube, stack and gripper quantities shared by the observation, reward and termination terms.

``cube_a`` is the red cube the robot picks up; ``cube_b`` is the blue cube it stacks it on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply

from .geometry import (
    CUBE_REST_HEIGHT,
    CUBE_SIZE,
    cube_flatness,
    cube_lowest_offset,
    horizontal_distance,
    stack_offset,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

FIXED_FINGERTIP_OFFSET = (-0.01289, -0.00022, -0.0986)
MOVING_FINGERTIP_OFFSET = (-0.0068, -0.0766, 0.0189)
"""Fingertip pad centers in the gripper and moving-jaw link frames [m] (same robot as the vial task)."""
GRASP_POINT_HEIGHT = 0.005
"""The jaws close this far above the red cube's center, so the fingertips stay above its bottom when stacking [m]."""

HOLD_HEIGHT = 0.006
"""A red cube this far above its resting height, with both jaws touching it, is held off the desk [m]."""
LIFT_CLEARANCE = 0.010
"""A red cube whose lowest corner is this far above the blue cube's top face has been lifted clear of it [m]."""
PLACED_HORIZONTAL, PLACED_VERTICAL = 0.015, 0.004
"""Stack tolerances for the placement milestone: the red center is over the blue top face [m]."""
SUCCESS_HORIZONTAL, SUCCESS_VERTICAL = 0.010, 0.004
"""Stack tolerances for success: the red center within 10 mm of centered, resting on the top face [m]."""
FLAT_ALIGNMENT = 0.966
"""Cosine of the most vertical face normal with world up for a flat cube; about 15 degrees of tilt."""
TOPPLED_ALIGNMENT = 0.906
"""A blue cube tilted beyond about 25 degrees has toppled."""
BLUE_HEIGHT_TOLERANCE = 0.008
"""A blue cube whose center leaves its resting height by more than this has been lifted or has fallen [m]."""
STILL_LINEAR, STILL_ANGULAR = 0.05, 0.5
"""Speed limits for a settled stack [m/s, rad/s]."""
RELEASED_OPENNESS = 0.2
"""Gripper openness, as a fraction of its range, above which the jaws count as released."""
HARD_STACK_IMPACT_FORCE = 20.0
"""Red/blue contact forces above this [N] count as an unsafe impact, logged but not terminated."""


def contact_force(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    """Largest contact force magnitude [N] of a sensor, filtered if the sensor declares filters."""
    sensor = env.scene.sensors[sensor_name]
    forces = sensor.data.force_matrix_w if sensor.cfg.filter_prim_paths_expr else sensor.data.net_forces_w
    return torch.linalg.vector_norm(forces.torch, dim=-1).reshape(env.num_envs, -1).amax(dim=-1)


def jaw_contacts(env: ManagerBasedRLEnv, threshold: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """Whether the fixed and the moving jaw touch something (the moving jaw sensor sees only the red cube)."""
    return contact_force(env, "fixed_jaw_contact") > threshold, contact_force(env, "moving_jaw_contact") > threshold


def fingertip_positions_w(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed and moving fingertip pad centers in the world frame [m]."""
    robot = env.scene["robot"]
    pads = []
    for body, offset in (("gripper", FIXED_FINGERTIP_OFFSET), ("moving_jaw_so101_v1", MOVING_FINGERTIP_OFFSET)):
        body_id = robot.find_bodies(body)[0][0]
        pos, quat = robot.data.body_pos_w.torch[:, body_id], robot.data.body_quat_w.torch[:, body_id]
        pads.append(pos + quat_apply(quat, pos.new_tensor(offset).expand_as(pos)))
    return pads[0], pads[1]


def grasp_center_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Midpoint between the two fingertip pads in the world frame [m]."""
    fixed, moving = fingertip_positions_w(env)
    return 0.5 * (fixed + moving)


def cube_grasp_point_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The point on the red cube the jaws close on, in the world frame [m]."""
    position = env.scene["cube_a"].data.root_pos_w.torch
    return position + position.new_tensor((0.0, 0.0, GRASP_POINT_HEIGHT))


def cube_position(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    """Cube center relative to the environment origin [m]."""
    return env.scene[name].data.root_pos_w.torch - env.scene.env_origins


def red_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Red cube center height above the environment origin [m]."""
    return cube_position(env, "cube_a")[:, 2]


def stack_state(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
    """Red-on-blue geometry: offset from the ideal stacked center, clearance, and both cubes' flatness."""
    red, blue = env.scene["cube_a"], env.scene["cube_b"]
    red_pos, blue_pos = red.data.root_pos_w.torch, blue.data.root_pos_w.torch
    red_quat, blue_quat = red.data.root_quat_w.torch, blue.data.root_quat_w.torch
    offset = stack_offset(red_pos, blue_pos)
    blue_top = blue_pos[:, 2] + CUBE_SIZE / 2
    return {
        "offset": offset,
        "horizontal": horizontal_distance(offset),
        "clearance": red_pos[:, 2] + cube_lowest_offset(red_quat) - blue_top,
        "red_flatness": cube_flatness(red_quat),
        "blue_flatness": cube_flatness(blue_quat),
        "blue_height": blue_pos[:, 2] - env.scene.env_origins[:, 2],
    }


def cube_speeds(env: ManagerBasedRLEnv, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear and angular speed of a cube [m/s, rad/s]."""
    data = env.scene[name].data
    return (
        torch.linalg.vector_norm(data.root_lin_vel_w.torch, dim=-1),
        torch.linalg.vector_norm(data.root_ang_vel_w.torch, dim=-1),
    )


def blue_on_desk(env: ManagerBasedRLEnv, state: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
    """The blue cube still rests flat on the desk (or mat), wherever it has been nudged."""
    state = stack_state(env) if state is None else state
    support_height = getattr(env, "_so101_support_height", 0.0)
    resting = (state["blue_height"] - CUBE_REST_HEIGHT - support_height).abs() < BLUE_HEIGHT_TOLERANCE
    return resting & (state["blue_flatness"] > TOPPLED_ALIGNMENT)


def gripper_openness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Measured jaw opening normalized to [0, 1] over the soft joint limits."""
    robot = env.scene["robot"]
    joint_id = robot.find_joints("gripper")[0][0]
    position = robot.data.joint_pos.torch[:, joint_id]
    lower, upper = robot.data.soft_joint_pos_limits.torch[:, joint_id].unbind(-1)
    return ((position - lower) / (upper - lower)).clamp(0.0, 1.0)
