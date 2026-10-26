"""Pure tensor geometry for cube placement in a four-cell square tray."""

import torch


def quat_conjugate_xyzw(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((-quat[..., :3], quat[..., 3:]), dim=-1)


def quat_rotate_xyzw(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quat[..., :3]
    return vector + 2.0 * (
        quat[..., 3:] * torch.cross(xyz, vector, dim=-1) + torch.cross(xyz, torch.cross(xyz, vector, dim=-1), dim=-1)
    )


TRAY_CELL_CENTERS = ((0.0, 0.0), (0.06, 0.0), (0.06, 0.06), (0.0, 0.06))


def tray_local_position(
    point_w: torch.Tensor, tray_pos_w: torch.Tensor, tray_quat_w: torch.Tensor
) -> torch.Tensor:
    return quat_rotate_xyzw(quat_conjugate_xyzw(tray_quat_w), point_w - tray_pos_w)


def cube_bottom_and_center_offset(
    cube_pos_w: torch.Tensor,
    cube_quat_w: torch.Tensor,
    tray_pos_w: torch.Tensor,
    tray_quat_w: torch.Tensor,
    half_side: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    signs = cube_pos_w.new_tensor(
        [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]
    )
    corners_w = cube_pos_w[:, None, :] + quat_rotate_xyzw(cube_quat_w[:, None, :], signs[None, :, :] * half_side)
    corners = tray_local_position(
        corners_w.reshape(-1, 3),
        tray_pos_w[:, None, :].expand(-1, 8, -1).reshape(-1, 3),
        tray_quat_w[:, None, :].expand(-1, 8, -1).reshape(-1, 4),
    ).reshape(-1, 8, 3)
    bottom = corners[..., 2].amin(dim=-1)
    center = tray_local_position(cube_pos_w, tray_pos_w, tray_quat_w)
    return bottom, center[..., :2], corners


def cube_in_tray(
    cube_pos_w: torch.Tensor,
    cube_quat_w: torch.Tensor,
    tray_pos_w: torch.Tensor,
    tray_quat_w: torch.Tensor,
    tray_bottom_height: float = 0.014,
    opening_half_side: float = 0.01125,
    half_side: float = 0.01,
) -> torch.Tensor:
    bottom, _, corners = cube_bottom_and_center_offset(cube_pos_w, cube_quat_w, tray_pos_w, tray_quat_w, half_side)
    cell_centers = corners.new_tensor(TRAY_CELL_CENTERS)
    xy = corners[..., :2]
    inside = ((xy[:, None, :, :] - cell_centers[None, :, None, :]).abs() <= opening_half_side).all(dim=-1).all(dim=-1)
    return ((bottom - tray_bottom_height).abs() <= 0.004) & inside.any(dim=-1)


def tray_placement_error(offset: torch.Tensor, bottom: torch.Tensor, tray_bottom_height: float = 0.014) -> torch.Tensor:
    cell_centers = offset.new_tensor(TRAY_CELL_CENTERS)
    horizontal = torch.linalg.vector_norm(offset[:, None, :] - cell_centers[None, :, :], dim=-1).amin(dim=-1)
    return horizontal + (bottom - tray_bottom_height).abs()


def inside_bounds(point: torch.Tensor, lower: tuple[float, ...], upper: tuple[float, ...]) -> torch.Tensor:
    return ((point >= point.new_tensor(lower)) & (point <= point.new_tensor(upper))).all(dim=-1)


def tabletop_cube_overlaps_tray(
    cube_pose: torch.Tensor, tray_pose: torch.Tensor, clearance: float = 0.0
) -> torch.Tensor:
    delta = cube_pose[:, :2] - tray_pose[:, :2]
    return torch.linalg.vector_norm(delta, dim=-1) < 0.04 + clearance
