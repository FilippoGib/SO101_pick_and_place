"""Pure tensor geometry of the two-cube stack, used by the environment, the reset generator and unit tests.

Quaternions are XYZW, matching Isaac Lab's tensors. Positions are relative to the environment origin unless a name
says otherwise.
"""

import torch

CUBE_SIZE = 0.03
"""Edge length of both cubes [m]."""
DESK_TOP_Z = 0.035
"""Height of the desk top (and of the sim2real mat at its lowest) above the environment origin [m]."""
CUBE_REST_HEIGHT = DESK_TOP_Z + CUBE_SIZE / 2
"""Center height of a cube resting on the desk [m]."""
STACK_HEIGHT = CUBE_SIZE
"""Center-to-center height of a cube stacked flat on another [m]."""


def quat_rotate_xyzw(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by XYZW quaternions; both broadcast over leading dimensions."""
    xyz, w = quat[..., :3], quat[..., 3:4]
    t = 2.0 * torch.linalg.cross(xyz, vector, dim=-1)
    return vector + w * t + torch.linalg.cross(xyz, t, dim=-1)


def cube_axes_z(quat_w: torch.Tensor) -> torch.Tensor:
    """World-z components of the cube's three local axes: (..., 3)."""
    eye = torch.eye(3, device=quat_w.device, dtype=quat_w.dtype)
    axes = quat_rotate_xyzw(quat_w.unsqueeze(-2), eye.expand(*quat_w.shape[:-1], 3, 3))
    return axes[..., 2]


def cube_flatness(quat_w: torch.Tensor) -> torch.Tensor:
    """Cosine between world up and the cube's most vertical face normal, in [1/sqrt(3), 1].

    A cube has no preferred up face, so any face pointing up counts as flat.
    """
    return cube_axes_z(quat_w).abs().amax(dim=-1)


def cube_lowest_offset(quat_w: torch.Tensor, size: float = CUBE_SIZE) -> torch.Tensor:
    """Height of the cube's lowest corner relative to its center (negative) [m]."""
    return -0.5 * size * cube_axes_z(quat_w).abs().sum(dim=-1)


def stack_offset(top_position: torch.Tensor, bottom_position: torch.Tensor) -> torch.Tensor:
    """Top-cube center relative to the ideal stacked center on the bottom cube: (..., 3) [m].

    The ideal center sits :data:`STACK_HEIGHT` straight above the bottom cube's center, so a flat, centered stack has
    a zero offset regardless of yaw.
    """
    ideal = bottom_position + bottom_position.new_tensor((0.0, 0.0, STACK_HEIGHT))
    return top_position - ideal


def horizontal_distance(offset: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(offset[..., :2], dim=-1)


def resting_on(
    top_position: torch.Tensor,
    bottom_position: torch.Tensor,
    horizontal_tolerance: float,
    vertical_tolerance: float,
) -> torch.Tensor:
    """Whether the top cube's center lies within the given tolerances of the ideal stacked center."""
    offset = stack_offset(top_position, bottom_position)
    return (horizontal_distance(offset) <= horizontal_tolerance) & (offset[..., 2].abs() <= vertical_tolerance)


def cubes_overlap_xy(position_a: torch.Tensor, position_b: torch.Tensor, min_gap: float) -> torch.Tensor:
    """Conservative tabletop overlap test for reset sampling: centers closer than ``min_gap`` [m] overlap.

    A cube's footprint fits in a circle of radius ``CUBE_SIZE / sqrt(2)``, so any gap above ``sqrt(2) * CUBE_SIZE``
    (42 mm) keeps two cubes apart whatever their yaw.
    """
    return horizontal_distance(position_a - position_b) < min_gap


def yaw_quat(angle: torch.Tensor) -> torch.Tensor:
    """XYZW quaternion of a rotation by ``angle`` about world z."""
    half = 0.5 * angle
    zeros = torch.zeros_like(angle)
    return torch.stack((zeros, zeros, half.sin(), half.cos()), dim=-1)
