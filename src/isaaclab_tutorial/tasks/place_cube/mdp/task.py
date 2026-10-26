"""Cube, cup, gripper, and contact quantities shared by task terms."""

import torch

from isaaclab.utils.math import quat_apply

from .geometry import cup_local_position, cup_upright, inside_cup_error

# The cube rests with its center at the tabletop pose used by the scene and
# reset dataset. This is also the reference height for detecting a drop.
CUBE_REST_HEIGHT = 0.06
CUBE_HALF_SIDE = 0.01
CUP_BOTTOM_HEIGHT = 0.004
CUP_RIM_HEIGHT = 0.080
CUP_INNER_HALF_WIDTH = 0.035
UPRIGHT_ALIGNMENT = 0.90
HARD_CUP_IMPACT_FORCE = 20.0
FIXED_FINGERTIP_OFFSET = (-0.01289, -0.00022, -0.0986)
MOVING_FINGERTIP_OFFSET = (-0.0068, -0.0766, 0.0189)


def contact_force(env, sensor_name: str) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    forces = sensor.data.force_matrix_w if sensor.cfg.filter_prim_paths_expr else sensor.data.net_forces_w
    return torch.linalg.vector_norm(forces.torch, dim=-1).reshape(env.num_envs, -1).amax(dim=-1)


def jaw_contacts(env, threshold: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    return contact_force(env, "fixed_jaw_contact") > threshold, contact_force(env, "moving_jaw_contact") > threshold


def fingertip_positions_w(env):
    robot = env.scene["robot"]
    pads = []
    for body, offset in (("gripper", FIXED_FINGERTIP_OFFSET), ("moving_jaw_so101_v1", MOVING_FINGERTIP_OFFSET)):
        body_id = robot.find_bodies(body)[0][0]
        pos = robot.data.body_pos_w.torch[:, body_id]
        quat = robot.data.body_quat_w.torch[:, body_id]
        pads.append(pos + quat_apply(quat, pos.new_tensor(offset).expand_as(pos)))
    return pads[0], pads[1]


def grasp_center_w(env):
    fixed, moving = fingertip_positions_w(env)
    return (fixed + moving) / 2


def cube_grasp_point_w(env):
    return env.scene["cube"].data.root_pos_w.torch


def cube_height(env):
    return env.scene["cube"].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]


def cube_in_cup(env):
    cube, cup = env.scene["cube"], env.scene["cup"]
    position = cup_local_position(cube.data.root_pos_w.torch, cup.data.root_pos_w.torch, cup.data.root_quat_w.torch)
    axis = cup_local_position(
        cube.data.root_pos_w.torch + cube.data.root_quat_w.torch.new_tensor((0.0, 0.0, 1.0)),
        cup.data.root_pos_w.torch,
        cup.data.root_quat_w.torch,
    ) - cup_local_position(cube.data.root_pos_w.torch, cup.data.root_pos_w.torch, cup.data.root_quat_w.torch)
    return position, axis, position[:, 2] - CUBE_HALF_SIDE


def cube_upright(env):
    return cup_upright(env.scene["cup"].data.root_quat_w.torch).float()


def cube_contained(env):
    from .geometry import cube_fully_inside_cup

    cube, cup = env.scene["cube"], env.scene["cup"]
    return cube_fully_inside_cup(
        cube.data.root_pos_w.torch,
        cube.data.root_quat_w.torch,
        cup.data.root_pos_w.torch,
        cup.data.root_quat_w.torch,
        CUP_INNER_HALF_WIDTH,
        CUP_BOTTOM_HEIGHT,
        CUP_RIM_HEIGHT,
        CUBE_HALF_SIDE,
    )


def gripper_openness(env):
    robot = env.scene["robot"]
    joint_id = robot.find_joints("gripper")[0][0]
    position = robot.data.joint_pos.torch[:, joint_id]
    lower, upper = robot.data.soft_joint_pos_limits.torch[:, joint_id].unbind(-1)
    return ((position - lower) / (upper - lower)).clamp(0.0, 1.0)


def cup_placement_error(env):
    return inside_cup_error(cube_in_cup(env)[0])
