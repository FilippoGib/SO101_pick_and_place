"""Cube, tray, gripper, and contact quantities shared by task terms."""

import torch

from isaaclab.utils.math import quat_apply

from .geometry import cube_bottom_and_center_offset, tray_placement_error

# The cube rests with its center at the tabletop pose used by the scene and
# reset dataset. This is also the reference height for detecting a drop.
CUBE_REST_HEIGHT = 0.06
CUBE_HALF_SIDE = 0.01
TRAY_OPENING_SIDE = 0.0225
TRAY_BOTTOM_HEIGHT = 0.014
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


def cube_on_tray(env):
    cube, tray = env.scene["cube"], env.scene["tray"]
    bottom, offset, _ = cube_bottom_and_center_offset(
        cube.data.root_pos_w.torch,
        cube.data.root_quat_w.torch,
        tray.data.root_pos_w.torch,
        tray.data.root_quat_w.torch,
        CUBE_HALF_SIDE,
    )
    return tray_placement_error(offset, bottom, TRAY_BOTTOM_HEIGHT)


def cube_supported(env):
    from .geometry import cube_in_tray as _cube_in_tray

    cube, tray = env.scene["cube"], env.scene["tray"]
    return _cube_in_tray(
        cube.data.root_pos_w.torch,
        cube.data.root_quat_w.torch,
        tray.data.root_pos_w.torch,
        tray.data.root_quat_w.torch,
        TRAY_BOTTOM_HEIGHT,
        TRAY_OPENING_SIDE / 2.0,
        CUBE_HALF_SIDE,
    )


def gripper_openness(env):
    robot = env.scene["robot"]
    joint_id = robot.find_joints("gripper")[0][0]
    position = robot.data.joint_pos.torch[:, joint_id]
    lower, upper = robot.data.soft_joint_pos_limits.torch[:, joint_id].unbind(-1)
    return ((position - lower) / (upper - lower)).clamp(0.0, 1.0)
