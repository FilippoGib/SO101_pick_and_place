"""Pure tensor geometry for cube placement inside an upright cup."""

import torch


def quat_conjugate_xyzw(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((-quat[..., :3], quat[..., 3:]), dim=-1)


def quat_rotate_xyzw(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quat[..., :3]
    return vector + 2.0 * (
        quat[..., 3:] * torch.cross(xyz, vector, dim=-1) + torch.cross(xyz, torch.cross(xyz, vector, dim=-1), dim=-1)
    )


def cup_local_position(point_w: torch.Tensor, cup_pos_w: torch.Tensor, cup_quat_w: torch.Tensor) -> torch.Tensor:
    return quat_rotate_xyzw(quat_conjugate_xyzw(cup_quat_w), point_w - cup_pos_w)


def cube_corners_in_cup(
    cube_pos_w: torch.Tensor,
    cube_quat_w: torch.Tensor,
    cup_pos_w: torch.Tensor,
    cup_quat_w: torch.Tensor,
    half_side: float = 0.01,
) -> torch.Tensor:
    signs = cube_pos_w.new_tensor(
        [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]
    )
    corners_w = cube_pos_w[:, None, :] + quat_rotate_xyzw(cube_quat_w[:, None, :], signs[None, :, :] * half_side)
    return cup_local_position(corners_w, cup_pos_w[:, None, :], cup_quat_w[:, None, :])


def cube_fully_inside_cup(
    cube_pos_w: torch.Tensor,
    cube_quat_w: torch.Tensor,
    cup_pos_w: torch.Tensor,
    cup_quat_w: torch.Tensor,
    inner_half_width: float = 0.035,
    bottom_height: float = 0.004,
    rim_height: float = 0.080,
    half_side: float = 0.01,
) -> torch.Tensor:
    corners = cube_corners_in_cup(cube_pos_w, cube_quat_w, cup_pos_w, cup_quat_w, half_side)
    return (
        (corners[..., 0].abs() <= inner_half_width).all(dim=-1)
        & (corners[..., 1].abs() <= inner_half_width).all(dim=-1)
        & (corners[..., 2] >= bottom_height).all(dim=-1)
        & (corners[..., 2] <= rim_height).all(dim=-1)
    )


def cup_upright(cup_quat_w: torch.Tensor, minimum_alignment: float = 0.9) -> torch.Tensor:
    up = cup_quat_w.new_tensor((0.0, 0.0, 1.0)).expand(cup_quat_w.shape[0], -1)
    return quat_rotate_xyzw(cup_quat_w, up)[..., 2] > minimum_alignment


def inside_cup_error(local_position: torch.Tensor, target_height: float = 0.035) -> torch.Tensor:
    return torch.linalg.vector_norm(local_position[..., :2], dim=-1) + (local_position[..., 2] - target_height).abs()


def inside_bounds(point: torch.Tensor, lower: tuple[float, ...], upper: tuple[float, ...]) -> torch.Tensor:
    return ((point >= point.new_tensor(lower)) & (point <= point.new_tensor(upper))).all(dim=-1)


def tabletop_cube_overlaps_cup(cube_pose: torch.Tensor, cup_pose: torch.Tensor, clearance: float = 0.0) -> torch.Tensor:
    delta = cube_pose[:, :2] - cup_pose[:, :2]
    return torch.linalg.vector_norm(delta, dim=-1) < 0.055 + clearance
