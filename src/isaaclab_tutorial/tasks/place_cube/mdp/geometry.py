"""Pure tensor geometry for cube placement on a cylindrical support."""

import torch


def quat_conjugate_xyzw(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((-quat[..., :3], quat[..., 3:]), dim=-1)


def quat_rotate_xyzw(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quat[..., :3]
    return vector + 2.0 * (
        quat[..., 3:] * torch.cross(xyz, vector, dim=-1) + torch.cross(xyz, torch.cross(xyz, vector, dim=-1), dim=-1)
    )


def cylinder_local_position(point_w: torch.Tensor, cylinder_pos_w: torch.Tensor) -> torch.Tensor:
    return point_w - cylinder_pos_w


def cube_bottom_and_center_offset(
    cube_pos_w: torch.Tensor,
    cube_quat_w: torch.Tensor,
    cylinder_pos_w: torch.Tensor,
    half_side: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    signs = cube_pos_w.new_tensor(
        [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]
    )
    corners_w = cube_pos_w[:, None, :] + quat_rotate_xyzw(cube_quat_w[:, None, :], signs[None, :, :] * half_side)
    bottom = corners_w[..., 2].amin(dim=-1)
    offset = cube_pos_w[:, :2] - cylinder_pos_w[:, :2]
    return bottom, offset


def cube_on_cylinder(
    cube_pos_w: torch.Tensor,
    cube_quat_w: torch.Tensor,
    cylinder_pos_w: torch.Tensor,
    cylinder_top_height: float = 0.045,
    cylinder_radius: float = 0.03,
    half_side: float = 0.01,
) -> torch.Tensor:
    bottom, offset = cube_bottom_and_center_offset(cube_pos_w, cube_quat_w, cylinder_pos_w, half_side)
    return ((bottom - cylinder_top_height).abs() <= 0.004) & (
        torch.linalg.vector_norm(offset, dim=-1) <= cylinder_radius - half_side
    )


def cylinder_placement_error(offset: torch.Tensor, bottom: torch.Tensor, top_height: float = 0.045) -> torch.Tensor:
    return torch.linalg.vector_norm(offset, dim=-1) + (bottom - top_height).abs()


def inside_bounds(point: torch.Tensor, lower: tuple[float, ...], upper: tuple[float, ...]) -> torch.Tensor:
    return ((point >= point.new_tensor(lower)) & (point <= point.new_tensor(upper))).all(dim=-1)


def tabletop_cube_overlaps_cylinder(
    cube_pose: torch.Tensor, cylinder_pose: torch.Tensor, clearance: float = 0.0
) -> torch.Tensor:
    delta = cube_pose[:, :2] - cylinder_pose[:, :2]
    return torch.linalg.vector_norm(delta, dim=-1) < 0.04 + clearance
