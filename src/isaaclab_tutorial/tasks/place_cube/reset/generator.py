"""Generate and inspect physics-validated reset poses for the cube-stacking task.

Every row is a simulated state reached through one connected physical rollout from the home pose: a top-down grasp of
the red cube, a lift, a carry to above the blue cube, a placement, and a release. Rows are kept only if their physical
state passes the checks for their phase, and the ``grasped``/``lifted`` flags come from the measured state, so the
milestones the environment restores on reset always match what the policy sees.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from isaaclab_tutorial.assets import CUBE_RESET_DATASET
from isaaclab_tutorial.tasks.place_cube.config.so101.env_cfg import (
    CUBE_A_POSITION,
    CUBE_B_POSITION,
    GRASP_GRIPPER_POSITION,
    PREGRASP_GRIPPER_POSITION,
    RELEASE_GRIPPER_POSITION,
    WORKSHOP_INITIAL_JOINT_POSITION,
)
from isaaclab_tutorial.tasks.place_cube.mdp.geometry import (
    CUBE_REST_HEIGHT,
    CUBE_SIZE,
    STACK_HEIGHT,
    cube_flatness,
    cubes_overlap_xy,
    horizontal_distance,
    quat_rotate_xyzw,
    stack_offset,
    yaw_quat,
)
from isaaclab_tutorial.tasks.place_cube.mdp.task import (
    FLAT_ALIGNMENT,
    GRASP_POINT_HEIGHT,
    HOLD_HEIGHT,
    LIFT_CLEARANCE,
    PLACED_HORIZONTAL,
    PLACED_VERTICAL,
    TOPPLED_ALIGNMENT,
    grasp_center_w,
    jaw_contacts,
)
from isaaclab_tutorial.tasks.place_cube.reset.dataset import PHASE_NAMES, load_reset_dataset, save_reset_dataset

HOME, APPROACH, GRASP, LIFT, TRANSPORT, PLACE, RELEASE = range(len(PHASE_NAMES))
LAST_PHASE = len(PHASE_NAMES)
"""Difficulty is ``(phase + fraction) / LAST_PHASE``, so the end of the release phase is 1."""
ROBOT_BASE_XY = (-0.05, 0.0)
"""Robot base position in the environment frame, used to pick the jaw orientation that faces the arm."""


def _is_finite_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class GeneratorCfg:
    """Reset generator quotas, layout ranges, and physical validation settings."""

    poses_per_phase: int = 128
    batch_size: int = 1024
    seed: int = 42
    articulation_settle_steps: int = 8
    grasp_close_steps: int = 90
    transition_steps: int = 120
    contact_settle_steps: int = 90
    max_attempts_per_phase: int = 32_768
    branch_seed_count: int = 32
    red_position_half_range: tuple[float, float] = (0.03, 0.03)
    red_yaw_range: tuple[float, float] = (-math.radians(30), math.radians(30))
    blue_position_half_range: tuple[float, float] = (0.015, 0.015)
    blue_yaw_range: tuple[float, float] = (-math.radians(15), math.radians(15))
    min_cube_gap: float = 0.06
    overhead_height: float = 0.05
    proof_lift: float = 0.020
    carry_height: float = 0.095
    grasp_pitch: float = math.radians(15.0)
    place_height: float = 0.002
    grasp_center_tolerance: float = 0.015
    blue_drift_tolerance: float = 0.010
    ik_seeds: int = 64
    ik_iterations: int = 120
    ik_noise_std: float = 0.65
    ik_position_tolerance: float = 0.005
    ik_rotation_tolerance: float = math.radians(12.0)
    ik_joint_margin: float = 0.03

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        for name in (
            "poses_per_phase",
            "batch_size",
            "articulation_settle_steps",
            "grasp_close_steps",
            "transition_steps",
            "contact_settle_steps",
            "max_attempts_per_phase",
            "branch_seed_count",
            "ik_seeds",
            "ik_iterations",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.max_attempts_per_phase < self.poses_per_phase:
            raise ValueError("max_attempts_per_phase must be at least poses_per_phase.")
        for name in (
            "min_cube_gap",
            "overhead_height",
            "proof_lift",
            "carry_height",
            "grasp_center_tolerance",
            "blue_drift_tolerance",
            "ik_position_tolerance",
            "ik_rotation_tolerance",
        ):
            value = getattr(self, name)
            if not _is_finite_number(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite number.")
        for name in ("place_height", "ik_noise_std", "ik_joint_margin"):
            value = getattr(self, name)
            if not _is_finite_number(value) or value < 0.0:
                raise ValueError(f"{name} must be a nonnegative finite number.")
        for name in ("red_position_half_range", "blue_position_half_range"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or len(value) != 2 or any(not _is_finite_number(v) or v < 0 for v in value):
                raise ValueError(f"{name} must contain two nonnegative half-widths.")
        for name in ("red_yaw_range", "blue_yaw_range"):
            value = getattr(self, name)
            if (
                not isinstance(value, tuple)
                or len(value) != 2
                or any(not _is_finite_number(v) for v in value)
                or not -math.pi <= value[0] <= value[1] <= math.pi
            ):
                raise ValueError(f"{name} must be an ordered finite interval within [-pi, pi].")
        if self.min_cube_gap < math.sqrt(2) * CUBE_SIZE:
            raise ValueError("min_cube_gap must keep two cubes apart at any yaw (at least sqrt(2) * CUBE_SIZE).")


def phase_quotas(cfg: GeneratorCfg) -> tuple[int, ...]:
    """Return the exact balanced phase quotas."""
    return (cfg.poses_per_phase,) * len(PHASE_NAMES)


def radial_direction(xy: torch.Tensor) -> torch.Tensor:
    """Horizontal unit vector from the robot base toward ``xy``: (..., 3)."""
    radial = xy - xy.new_tensor(ROBOT_BASE_XY)
    radial = radial / torch.linalg.vector_norm(radial, dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.nn.functional.pad(radial, (0, 1))


def pan_angle(xy: torch.Tensor) -> torch.Tensor:
    """Azimuth of ``xy`` about the robot base [rad]."""
    return torch.atan2(xy[..., 1] - ROBOT_BASE_XY[1], xy[..., 0] - ROBOT_BASE_XY[0])


def top_down_grasp_frame(red_quat: torch.Tensor, red_xy: torch.Tensor, pitch: float) -> torch.Tensor:
    """XYZW gripper orientation for a near-top-down grasp of a flat cube.

    Gripper +Z is the up axis (the fingers point along -Z), +X is the closing direction and +Y runs along the pads.
    Straight down is out of the SO-101's wrist range at these positions, so +Z is tilted by ``pitch`` within the
    vertical plane through the robot base, the only plane a five-DoF arm can tilt in at any wrist roll. The jaws close
    along the cube face normal nearest the horizontal direction across that plane, projected square to the tilted
    gripper, so the pads stay parallel to a pair of faces up to the pitch.
    """
    from isaaclab.utils.math import quat_from_angle_axis, quat_from_matrix

    count = red_quat.shape[0]
    radial = radial_direction(red_xy)
    world_up = torch.zeros_like(radial)
    world_up[:, 2] = 1.0
    lateral = torch.linalg.cross(radial, world_up, dim=-1)
    up = quat_rotate_xyzw(quat_from_angle_axis(torch.full((count,), pitch, device=red_quat.device), lateral), world_up)
    eye = torch.eye(3, device=red_quat.device, dtype=red_quat.dtype)
    face_axes = quat_rotate_xyzw(red_quat.unsqueeze(1), eye[:2].expand(count, 2, 3))
    candidates = torch.cat((face_axes, -face_axes), dim=1)
    normal = candidates[torch.arange(count), (candidates * lateral[:, None]).sum(-1).argmax(-1)]
    closing = normal - (normal * up).sum(-1, keepdim=True) * up
    closing = closing / torch.linalg.vector_norm(closing, dim=-1, keepdim=True).clamp_min(1e-6)
    pad = torch.linalg.cross(up, closing, dim=-1)
    return quat_from_matrix(torch.stack((closing, pad, up), dim=-1))


def _tensor(value):
    return value.torch if hasattr(value, "torch") else value


class _Generator:
    """Build candidates in simulation and keep only stable physical states."""

    def __init__(self, env, cfg: GeneratorCfg):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(env.device)
        self.random = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.robot = env.scene["robot"]
        self.red = env.scene["cube_a"]
        self.blue = env.scene["cube_b"]
        self.num_envs = env.num_envs
        self.gripper_body_id = self.robot.find_bodies("gripper", preserve_order=True)[0]
        self._zeros = torch.zeros((self.num_envs, 6), device=self.device)
        self.rejections = {name: 0 for name in PHASE_NAMES}
        self.last_diagnostics: dict[str, float] = {}
        self.candidate_diagnostics: dict[str, float] = {}
        self.candidate_difficulty = torch.zeros(self.num_envs, device=self.device)
        self.candidate_ik_valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._seed_banks: dict[str, dict[str, torch.Tensor]] = {}
        self.red_start = torch.zeros((self.num_envs, 7), device=self.device)
        self.blue_start = torch.zeros((self.num_envs, 7), device=self.device)
        # Pose the gripper open before measuring the aperture center; the env has not been reset yet.
        open_geometry_position = _tensor(self.robot.data.default_joint_pos).clone()
        open_geometry_position[:, -1] = PREGRASP_GRIPPER_POSITION
        self._write_robot(open_geometry_position)
        self.env.scene.write_data_to_sim()
        self.env.sim.forward()
        self.env.scene.update(self.env.physics_dt)
        self.grasp_tcp_offset = self._pregrasp_center_offset()
        self._build_ik()
        gripper_joint_id = self.robot.find_joints("gripper", preserve_order=True)[0][0]
        lower, upper = self.ik_joint_limits[gripper_joint_id].tolist()
        for name, value in (("grasp", GRASP_GRIPPER_POSITION), ("release", RELEASE_GRIPPER_POSITION)):
            if not lower <= value <= upper:
                raise RuntimeError(
                    f"The real-robot {name} command {value:.7f} rad is outside the USD soft limits "
                    f"[{lower:.7f}, {upper:.7f}]."
                )

    # --- geometry and IK -------------------------------------------------------------------------------------------

    def _pregrasp_center_offset(self) -> torch.Tensor:
        """Measure the open-jaw pad midpoint in the gripper frame, used to center the jaws on the cube."""
        from isaaclab.utils.math import quat_apply_inverse

        position = _tensor(self.robot.data.body_pos_w)[:, self.gripper_body_id].squeeze(1)
        quaternion = _tensor(self.robot.data.body_quat_w)[:, self.gripper_body_id].squeeze(1)
        offsets = quat_apply_inverse(quaternion, grasp_center_w(self.env) - position)
        if not bool(torch.isfinite(offsets).all()):
            raise RuntimeError("The USD-authored pregrasp geometry is not finite.")
        if float((offsets - offsets[0]).abs().amax()) > 1.0e-5:
            raise RuntimeError("Cloned environments disagree on the pregrasp contact geometry.")
        return offsets[0].clone()

    def _build_ik(self) -> None:
        """Build the batched Newton IK used by the vial task's reset generator."""
        import warp as wp

        import isaaclab.sim as sim_utils
        from isaaclab import cloner
        from isaaclab_newton.cloner import copy_newton_clone_source
        from isaaclab_newton.controllers.ik import (
            NewtonIKJointLimitObjectiveCfg,
            NewtonIKPoseObjectiveCfg,
            NewtonIKSolver,
            NewtonIKSolverCfg,
        )

        plan = sim_utils.SimulationContext.instance().get_clone_plan()
        if plan is None:
            raise RuntimeError("Reset IK requires a Newton clone plan.")
        asset_ids = cloner.path.get_asset_prototypes(plan, self.robot.cfg.prim_path)
        sources = cloner.path.get_asset_prototype_paths(plan)
        source_paths = [sources[index] for index in asset_ids if sources[index] is not None]
        if len(source_paths) != 1:
            raise RuntimeError("Expected exactly one SO-101 clone source for reset IK.")
        source = copy_newton_clone_source(source_paths[0])
        origin = -self.env.scene.env_origins[0]
        prototype_xform = wp.transform(wp.vec3(*origin.tolist()), wp.quat_identity())
        import newton

        prototype = newton.ModelBuilder(up_axis=source.up_axis)
        prototype.add_builder(source, xform=prototype_xform)
        self.ik_model = prototype.finalize(device=str(self.device))

        body_names = [str(label).rsplit("/", 1)[-1] for label in self.ik_model.body_label]
        gripper_matches = [index for index, name in enumerate(body_names) if name == "gripper"]
        if len(gripper_matches) != 1:
            raise RuntimeError("Expected exactly one Newton IK body named 'gripper'.")
        self.ik_gripper_body_id = gripper_matches[0]

        joint_names = [str(label).rsplit("/", 1)[-1] for label in self.ik_model.joint_label]
        joint_q_start = wp.to_torch(self.ik_model.joint_q_start).to(device=self.device, dtype=torch.long)
        coordinate_ids = []
        for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"):
            matches = [index for index, joint_name in enumerate(joint_names) if joint_name == name]
            if len(matches) != 1:
                raise RuntimeError(f"Expected exactly one Newton IK joint named {name!r}.")
            coordinate_ids.append(int(joint_q_start[matches[0]].item()))
        self.ik_coordinate_ids = torch.tensor(coordinate_ids, device=self.device, dtype=torch.long)
        self.ik_joint_seed = wp.to_torch(self.ik_model.joint_q).to(self.device).repeat(self.num_envs, 1)
        self.ik_joint_limits = _tensor(self.robot.data.soft_joint_pos_limits)[0].clone()

        objective_name = "reset_cube_gripper"
        self.ik_solver = NewtonIKSolver(
            NewtonIKSolverCfg(
                optimizer="lm",
                jacobian_mode="analytic",
                sampler="gauss",
                n_seeds=self.cfg.ik_seeds,
                noise_std=self.cfg.ik_noise_std,
                iterations=self.cfg.ik_iterations,
                lambda_initial=0.1,
                rng_seed=self.cfg.seed,
            ),
            model=self.ik_model,
            num_envs=self.num_envs,
            device=str(self.device),
            objectives=[
                NewtonIKPoseObjectiveCfg(
                    body_name="gripper",
                    name=objective_name,
                    body_offset_pos=tuple(float(value) for value in self.grasp_tcp_offset),
                    position_weight=100.0,
                    # Five-DoF arm: position dominates, rotation keeps the jaws square to the cube.
                    rotation_weight=1.0,
                ),
                NewtonIKJointLimitObjectiveCfg(weight=1.0),
            ],
            link_resolver=lambda _name: self.ik_gripper_body_id,
        )
        self.ik_pose_objective = self.ik_solver.objectives_by_name[objective_name]

    def _solve_ik(self, target_position: torch.Tensor, target_quaternion: torch.Tensor, seed: torch.Tensor):
        """Run the multi-start IK and return joint solutions, costs and gripper poses per seed."""
        import warp as wp

        self.ik_pose_objective.position_objective.set_target_positions(
            wp.from_torch(target_position.contiguous(), dtype=wp.vec3)
        )
        self.ik_pose_objective.rotation_objective.set_target_rotations(
            wp.from_torch(target_quaternion.contiguous(), dtype=wp.vec4)
        )
        full_seed = self.ik_joint_seed.clone()
        full_seed[:, self.ik_coordinate_ids] = seed
        self.ik_solver.solve(wp.from_torch(full_seed.contiguous(), dtype=wp.float32))
        seeds = self.cfg.ik_seeds
        joint_q = wp.to_torch(self.ik_solver.joint_q).reshape(self.num_envs, seeds, -1)
        costs = wp.to_torch(self.ik_solver.costs).reshape(self.num_envs, seeds)
        body_pose = wp.to_torch(self.ik_solver.solver.body_q).reshape(self.num_envs, seeds, -1, 7)
        return joint_q, costs, body_pose[:, :, self.ik_gripper_body_id]

    def _tcp(self, gripper_pose: torch.Tensor) -> torch.Tensor:
        """TCP (open-jaw pad midpoint) position of gripper poses (..., 7)."""
        offset = self.grasp_tcp_offset.expand(*gripper_pose.shape[:-1], 3)
        return gripper_pose[..., :3] + quat_rotate_xyzw(gripper_pose[..., 3:7], offset)

    def _live_tcp(self) -> torch.Tensor:
        position = _tensor(self.robot.data.body_pos_w)[:, self.gripper_body_id].squeeze(1)
        quaternion = _tensor(self.robot.data.body_quat_w)[:, self.gripper_body_id].squeeze(1)
        return self._tcp(torch.cat((position, quaternion), dim=-1))

    def _select(self, joint_q, seed_valid, score, fallback_joint_position, gripper_position):
        """Pick the best valid seed per row; rejected rows keep their finite fallback pose."""
        valid = seed_valid.any(-1)
        fallback = torch.nan_to_num(score, nan=torch.inf, posinf=torch.inf, neginf=torch.inf).argmin(-1)
        selected = torch.where(valid, score.masked_fill(~seed_valid, torch.inf).argmin(-1), fallback)
        rows = torch.arange(self.num_envs, device=self.device)
        solved = joint_q[rows, selected][:, self.ik_coordinate_ids].clone()
        solved[~valid] = fallback_joint_position[~valid]
        solved[:, -1] = gripper_position
        self.candidate_ik_valid &= valid
        return solved, valid

    def _solve_top_down(
        self, target_position: torch.Tensor, target_quaternion: torch.Tensor, seed: torch.Tensor, prefix: str
    ) -> torch.Tensor:
        """Solve an open-jaw pose with the TCP at ``target_position`` and the gripper at ``target_quaternion``."""
        joint_q, costs, gripper_pose = self._solve_ik(target_position, target_quaternion, seed)
        position_error = torch.linalg.vector_norm(self._tcp(gripper_pose) - target_position[:, None], dim=-1)
        eye = gripper_pose.new_tensor(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        actual = quat_rotate_xyzw(gripper_pose[:, :, None, 3:7], eye.expand(*gripper_pose.shape[:2], 2, 3))
        desired = quat_rotate_xyzw(target_quaternion[:, None, :], eye.expand(self.num_envs, 2, 3))
        closing_alignment = (actual[:, :, 0] * desired[:, None, 0]).sum(-1)
        up_alignment = (actual[:, :, 1] * desired[:, None, 1]).sum(-1)
        arm_q = joint_q[:, :, self.ik_coordinate_ids[:5]]
        arm_limits = self.ik_joint_limits[:5]
        margin = torch.minimum(arm_q - arm_limits[:, 0], arm_limits[:, 1] - arm_q).amin(-1)
        cos_tolerance = math.cos(self.cfg.ik_rotation_tolerance)
        seed_valid = (
            torch.isfinite(joint_q).all(-1)
            & torch.isfinite(costs)
            & (position_error <= self.cfg.ik_position_tolerance)
            & (closing_alignment >= cos_tolerance)
            & (up_alignment >= cos_tolerance)
            & (margin >= self.cfg.ik_joint_margin)
        )
        continuity = torch.linalg.vector_norm(arm_q - seed[:, None, :5], dim=-1)
        score = (1.0 - closing_alignment) + (1.0 - up_alignment) + 0.05 * continuity + 10.0 * position_error
        solved, valid = self._select(joint_q, seed_valid, score, seed, PREGRASP_GRIPPER_POSITION)
        # Which arm joint limits the best-margin solution of each row, for diagnosing unreachable layouts.
        joint_margin = torch.minimum(arm_q - arm_limits[:, 0], arm_limits[:, 1] - arm_q)
        best = joint_margin[torch.arange(self.num_envs, device=self.device), margin.argmax(-1)]
        limited = best.amin(-1) < self.cfg.ik_joint_margin
        for index, name in enumerate(("pan", "lift", "elbow", "wrist_flex", "wrist_roll")):
            hits = limited & (best.argmin(-1) == index)
            self.candidate_diagnostics[f"{prefix}_limited_by_{name}_rate"] = float(hits.float().mean())
        self.candidate_diagnostics.update(
            {
                f"{prefix}_ik_valid_rate": float(valid.float().mean()),
                f"{prefix}_position_error_mean_m": float(position_error.amin(-1).mean()),
                f"{prefix}_closing_alignment_mean": float(closing_alignment.amax(-1).mean()),
                f"{prefix}_up_alignment_mean": float(up_alignment.amax(-1).mean()),
                f"{prefix}_margin_mean_rad": float(margin.amax(-1).mean()),
            }
        )
        return solved

    def _solve_held_pose(
        self, desired_red_pose: torch.Tensor, seed: torch.Tensor, position_tolerance: float, max_joint_delta: float
    ) -> torch.Tensor:
        """Solve the gripper pose that puts the held red cube at a desired pose, keeping its live offset in the jaws."""
        from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

        red_position = _tensor(self.red.data.root_pos_w)
        red_quaternion = _tensor(self.red.data.root_quat_w)
        gripper_position = _tensor(self.robot.data.body_pos_w)[:, self.gripper_body_id].squeeze(1)
        gripper_quaternion = _tensor(self.robot.data.body_quat_w)[:, self.gripper_body_id].squeeze(1)
        tcp_position = self._tcp(torch.cat((gripper_position, gripper_quaternion), dim=-1))
        red_to_tcp_position, red_to_tcp_quaternion = subtract_frame_transforms(
            red_position, red_quaternion, tcp_position, gripper_quaternion
        )
        # The IK model is the environment-0 robot moved to the origin, so targets are in the environment frame.
        target_position, target_quaternion = combine_frame_transforms(
            desired_red_pose[:, :3], desired_red_pose[:, 3:7], red_to_tcp_position, red_to_tcp_quaternion
        )
        joint_q, costs, gripper_pose = self._solve_ik(target_position, target_quaternion, seed)
        tcp_error = torch.linalg.vector_norm(self._tcp(gripper_pose) - target_position[:, None], dim=-1)
        # The held cube must stay flat: compare the solved gripper's up axis with the target's.
        up = gripper_pose.new_tensor((0.0, 0.0, 1.0))
        actual_up = quat_rotate_xyzw(gripper_pose[..., 3:7], up.expand(*gripper_pose.shape[:2], 3))
        target_up = quat_rotate_xyzw(target_quaternion, up.expand(self.num_envs, 3))
        up_alignment = (actual_up * target_up[:, None]).sum(-1)
        arm_q = joint_q[:, :, self.ik_coordinate_ids[:5]]
        arm_limits = self.ik_joint_limits[:5]
        margin = torch.minimum(arm_q - arm_limits[:, 0], arm_limits[:, 1] - arm_q).amin(-1)
        continuity = torch.linalg.vector_norm(arm_q - seed[:, None, :5], dim=-1)
        seed_valid = (
            torch.isfinite(joint_q).all(-1)
            & torch.isfinite(costs)
            & (tcp_error <= position_tolerance)
            & (up_alignment >= math.cos(self.cfg.ik_rotation_tolerance))
            & (margin >= self.cfg.ik_joint_margin)
            & (continuity <= max_joint_delta)
        )
        finite = torch.isfinite(joint_q).all(-1) & torch.isfinite(costs)
        for name, check in (
            ("position", tcp_error <= position_tolerance),
            ("orientation", up_alignment >= math.cos(self.cfg.ik_rotation_tolerance)),
            ("margin", margin >= self.cfg.ik_joint_margin),
            ("continuity", continuity <= max_joint_delta),
        ):
            self.candidate_diagnostics[f"held_{name}_ok_rate"] = float((finite & check).any(-1).float().mean())
        self.candidate_diagnostics.update(
            {
                "held_ik_valid_rate": float(seed_valid.any(-1).float().mean()),
                "held_tcp_error_mean_m": float(tcp_error.amin(-1).mean()),
                "held_joint_delta_mean_rad": float(continuity.amin(-1).mean()),
            }
        )
        # Stay on the current IK branch to avoid wrist flips while loaded.
        solved, _ = self._select(joint_q, seed_valid, continuity, seed, GRASP_GRIPPER_POSITION)
        solved[:, -1] = seed[:, -1]
        return solved

    # --- simulation helpers -----------------------------------------------------------------------------------------

    def _simulate(self, steps: int, target: torch.Tensor) -> None:
        for _ in range(steps):
            self.robot.set_joint_position_target_index(target=target)
            self.robot.set_joint_velocity_target_index(target=self._zeros)
            self.env.scene.write_data_to_sim()
            self.env.sim.step()
            self.env.scene.update(self.env.physics_dt)

    def _move_robot(self, target: torch.Tensor, steps: int) -> None:
        """Move to a joint waypoint smoothly while both cubes remain dynamic."""
        start = _tensor(self.robot.data.joint_pos).clone()
        for step in range(1, steps + 1):
            self._simulate(1, torch.lerp(start, target, step / steps))

    def _write_robot(self, joint_position: torch.Tensor) -> None:
        self.robot.write_joint_position_to_sim_index(position=joint_position)
        self.robot.write_joint_velocity_to_sim_index(velocity=self._zeros)
        self.robot.set_joint_position_target_index(target=joint_position)

    def _write_cube(self, cube, local_pose: torch.Tensor) -> None:
        world_pose = local_pose.clone()
        world_pose[:, :3] += self.env.scene.env_origins
        cube.write_root_pose_to_sim_index(root_pose=world_pose)
        cube.write_root_velocity_to_sim_index(root_velocity=self._zeros)

    def _cube_pose(self, cube) -> torch.Tensor:
        return torch.cat(
            (_tensor(cube.data.root_pos_w) - self.env.scene.env_origins, _tensor(cube.data.root_quat_w)), dim=-1
        )

    def _commit_reset_state(self) -> None:
        """Write pending state and refresh kinematics so rejected candidates leave no stale buffers."""
        self.env.scene.reset()
        self.env.scene.write_data_to_sim()
        self.env.sim.forward()
        self.env.scene.update(self.env.physics_dt)

    def _home_joint_positions(self) -> torch.Tensor:
        """The exact operational pose used to begin every full task, with the jaws open."""
        base = self._zeros.new_tensor(WORKSHOP_INITIAL_JOINT_POSITION).repeat(self.num_envs, 1)
        base[:, -1] = PREGRASP_GRIPPER_POSITION
        limits = _tensor(self.robot.data.soft_joint_pos_limits)
        return base.clamp(limits[..., 0], limits[..., 1])

    def _uniform(self, shape, low, high) -> torch.Tensor:
        return torch.empty(shape, device=self.device).uniform_(0.0, 1.0, generator=self.random) * (high - low) + low

    def sample_layout(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample tabletop poses of both cubes within the configured ranges, keeping them apart."""
        n = self.num_envs

        def pose(nominal, half_range, yaw_range):
            out = torch.zeros((n, 7), device=self.device)
            out[:, :3] = out.new_tensor(nominal)
            half = out.new_tensor(half_range)
            out[:, :2] += self._uniform((n, 2), -1.0, 1.0) * half
            out[:, 3:7] = yaw_quat(self._uniform((n,), *yaw_range))
            return out

        blue = pose(CUBE_B_POSITION, self.cfg.blue_position_half_range, self.cfg.blue_yaw_range)
        red = pose(CUBE_A_POSITION, self.cfg.red_position_half_range, self.cfg.red_yaw_range)
        for _ in range(64):
            redraw = cubes_overlap_xy(red[:, :3], blue[:, :3], self.cfg.min_cube_gap)
            if not redraw.any():
                break
            red[redraw] = pose(CUBE_A_POSITION, self.cfg.red_position_half_range, self.cfg.red_yaw_range)[redraw]
        else:
            raise RuntimeError("Could not sample non-overlapping cube layouts; widen the ranges or the nominal gap.")
        return red, blue

    def _start_batch(self) -> torch.Tensor:
        """Place the robot at home and settle a fresh cube layout on the desk; return the home command."""
        home = self._home_joint_positions()
        red, blue = self.sample_layout()
        self._write_robot(home)
        self._write_cube(self.red, red)
        self._write_cube(self.blue, blue)
        self._commit_reset_state()
        self._simulate(self.cfg.contact_settle_steps, home)
        self.red_start.copy_(self._cube_pose(self.red))
        self.blue_start.copy_(self._cube_pose(self.blue))
        return home

    def _grasp_targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        """TCP position (environment frame, like every IK target) and gripper orientation to close on the red cube."""
        position = self._cube_pose(self.red)[:, :3].clone()
        position[:, 2] += GRASP_POINT_HEIGHT
        return position, top_down_grasp_frame(self.red_start[:, 3:7], self.red_start[:, :2], self.cfg.grasp_pitch)

    def _approach_commands(self, home: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve the overhead and pregrasp commands for the settled layout."""
        grasp_position, grasp_quaternion = self._grasp_targets()
        overhead_position = grasp_position.clone()
        overhead_position[:, 2] += self.cfg.overhead_height
        overhead = self._solve_top_down(overhead_position, grasp_quaternion, home, "overhead")
        pregrasp = self._solve_top_down(grasp_position, grasp_quaternion, overhead, "pregrasp")
        return overhead, pregrasp

    def _center_open_gripper(self, target: torch.Tensor, corrections: int = 2) -> torch.Tensor:
        """Center the open jaws on the cube with bounded Cartesian feedback on the joint targets."""
        grasp_position, grasp_quaternion = self._grasp_targets()
        bias = torch.zeros((self.num_envs, 3), device=self.device)
        for _ in range(corrections):
            bias -= self._live_tcp() - self.env.scene.env_origins - grasp_position
            bias *= (0.02 / torch.linalg.vector_norm(bias, dim=-1, keepdim=True).clamp_min(0.02)).clamp_max(1.0)
            target = self._solve_top_down(grasp_position + bias, grasp_quaternion, target, "centering")
            self._move_robot(target, max(1, self.cfg.transition_steps // 2))
            self._simulate(max(1, self.cfg.contact_settle_steps // 3), target)
        error = self._live_tcp() - self.env.scene.env_origins - grasp_position
        self.candidate_diagnostics["centering_error_mean_m"] = float(torch.linalg.vector_norm(error, dim=-1).mean())
        self.candidate_diagnostics["centering_error_z_mean_m"] = float(error[:, 2].mean())
        return target

    def _carry(
        self, target: torch.Tensor, translation: torch.Tensor, progress: torch.Tensor, corrections: int = 2
    ) -> torch.Tensor:
        """Translate the held red cube through Cartesian waypoints, re-solving IK against its live pose.

        The position-controlled arm sags under gravity and load, so the measured cube error is fed back into the next
        IK target (bounded to 3 cm), and ``corrections`` extra steps close the remaining error at the end.
        """
        from isaaclab.utils.math import quat_mul

        start_pose = self._cube_pose(self.red).clone()
        settle_steps = max(1, self.cfg.contact_settle_steps // 6)
        connected_valid = self.candidate_ik_valid.clone()
        bias = torch.zeros((self.num_envs, 3), device=self.device)
        waypoints = (0.25, 0.50, 0.75, *([1.0] * (1 + corrections)))
        for index, waypoint in enumerate(waypoints):
            correcting = index >= len(waypoints) - corrections
            waypoint_progress = torch.minimum(progress, torch.full_like(progress, waypoint))
            desired = start_pose.clone()
            desired[:, :3] += translation * waypoint_progress.unsqueeze(-1)
            # Turn with the shoulder pan: a five-DoF arm keeps its wrist tilt in the plane through the base.
            turn = pan_angle(desired[:, :2]) - pan_angle(start_pose[:, :2])
            desired[:, 3:7] = quat_mul(yaw_quat(turn), start_pose[:, 3:7])
            command = desired.clone()
            command[:, :3] += bias
            self.candidate_ik_valid.fill_(True)
            proposed = self._solve_held_pose(command, target, position_tolerance=0.010, max_joint_delta=0.45)
            update = connected_valid & self.candidate_ik_valid
            # A failed correction keeps the last command; the physical checks still judge the row.
            if not correcting:
                connected_valid &= self.candidate_ik_valid
            target = torch.where(update.unsqueeze(-1), proposed, target)
            self._move_robot(target, self.cfg.transition_steps)
            self._simulate(settle_steps, target)
            bias += desired[:, :3] - self._cube_pose(self.red)[:, :3]
            bias *= (0.03 / torch.linalg.vector_norm(bias, dim=-1, keepdim=True).clamp_min(0.03)).clamp_max(1.0)
        error = torch.linalg.vector_norm(desired[:, :3] - self._cube_pose(self.red)[:, :3], dim=-1)
        self.candidate_diagnostics["carry_error_mean_m"] = float(error.mean())
        self.candidate_diagnostics["carry_error_max_m"] = float(error.max())
        self.candidate_ik_valid.copy_(connected_valid)
        return target

    # --- seed banks ---------------------------------------------------------------------------------------------------

    def _append_seed_bank(self, name: str, valid: torch.Tensor, target: torch.Tensor) -> None:
        """Cache dynamically reached states so a later phase branches from real physics."""
        if not bool(valid.any()):
            return
        part = {
            "joint_position": _tensor(self.robot.data.joint_pos)[valid].detach().clone(),
            "joint_target": target[valid].detach().clone(),
            "cube_a_pose": self._cube_pose(self.red)[valid].detach().clone(),
            "cube_b_pose": self._cube_pose(self.blue)[valid].detach().clone(),
            "blue_start": self.blue_start[valid].detach().clone(),
        }
        if name in self._seed_banks:
            part = {key: torch.cat((self._seed_banks[name][key], value), dim=0) for key, value in part.items()}
        maximum = self.cfg.batch_size * 4
        self._seed_banks[name] = {key: value[-maximum:] for key, value in part.items()}

    def _restore_seed_bank(self, name: str) -> torch.Tensor:
        """Restore validated states across the batch and let them settle; return their joint targets."""
        bank = self._seed_banks[name]
        count = bank["joint_position"].shape[0]
        rows = torch.randint(count, (self.num_envs,), device=self.device, generator=self.random)
        target = bank["joint_target"][rows].clone()
        self._write_robot(bank["joint_position"][rows])
        self.robot.set_joint_position_target_index(target=target)
        self._write_cube(self.red, bank["cube_a_pose"][rows])
        self._write_cube(self.blue, bank["cube_b_pose"][rows])
        self.blue_start.copy_(bank["blue_start"][rows])
        self._commit_reset_state()
        self._simulate(max(1, self.cfg.contact_settle_steps // 3), target)
        return target

    def _segment_fraction(self, terminal_share: float = 0.5) -> torch.Tensor:
        """Uniform progress through a segment, with a share of rows at its end to seed the next phase."""
        fraction = self._uniform((self.num_envs,), 0.02, 0.98)
        terminal = torch.rand(self.num_envs, device=self.device, generator=self.random) < terminal_share
        terminal[-1] = True
        fraction[terminal] = 1.0
        return fraction

    # --- validity -----------------------------------------------------------------------------------------------------

    def _valid(self, phase: int) -> torch.Tensor:
        """Physical checks every row of ``phase`` must pass, with diagnostics of each check."""
        joint_position = _tensor(self.robot.data.joint_pos)
        joint_velocity = _tensor(self.robot.data.joint_vel)
        red, blue = self._cube_pose(self.red), self._cube_pose(self.blue)
        red_speed = torch.linalg.vector_norm(_tensor(self.red.data.root_lin_vel_w), dim=-1)
        red_spin = torch.linalg.vector_norm(_tensor(self.red.data.root_ang_vel_w), dim=-1)
        blue_speed = torch.linalg.vector_norm(_tensor(self.blue.data.root_lin_vel_w), dim=-1)
        finite = torch.isfinite(joint_position).all(-1) & torch.isfinite(joint_velocity).all(-1)
        finite &= torch.isfinite(red).all(-1) & torch.isfinite(blue).all(-1) & self.candidate_ik_valid
        limits = _tensor(self.robot.data.soft_joint_pos_limits)
        within_limits = ((joint_position >= limits[..., 0]) & (joint_position <= limits[..., 1])).all(-1)
        calm = joint_velocity.abs().amax(-1) < 2.0
        blue_drift = horizontal_distance(blue[:, :3] - self.blue_start[:, :3])
        blue_ok = (blue_drift < self.cfg.blue_drift_tolerance) & (cube_flatness(blue[:, 3:7]) > TOPPLED_ALIGNMENT)
        blue_ok &= ((blue[:, 2] - CUBE_REST_HEIGHT).abs() < 0.004) & (blue_speed < 0.08)
        fixed, moving = jaw_contacts(self.env)
        bilateral = fixed & moving
        grasp_point = _tensor(self.red.data.root_pos_w).clone()
        grasp_point[:, 2] += GRASP_POINT_HEIGHT
        centered = torch.linalg.vector_norm(grasp_center_w(self.env) - grasp_point, dim=-1)
        centered = centered < self.cfg.grasp_center_tolerance
        red_flat = cube_flatness(red[:, 3:7]) > FLAT_ALIGNMENT
        offset = stack_offset(red[:, :3], blue[:, :3])
        held = bilateral & centered & (red_speed < 0.15) & (red_spin < 1.0)
        held &= red[:, 2] > CUBE_REST_HEIGHT + HOLD_HEIGHT

        if phase in (HOME, APPROACH):
            red_drift = horizontal_distance(red[:, :3] - self.red_start[:, :3])
            phase_ok = (red_drift < 0.006) & ((red[:, 2] - CUBE_REST_HEIGHT).abs() < 0.004) & red_flat
            phase_ok &= (red_speed < 0.08) & ~(fixed | moving)
        elif phase == RELEASE:
            phase_ok = (horizontal_distance(offset) < PLACED_HORIZONTAL) & (offset[:, 2].abs() <= PLACED_VERTICAL)
            phase_ok &= red_flat & (red_speed < 0.08) & (red_spin < 1.0)
        else:
            phase_ok = held & red_flat
            if phase >= TRANSPORT:
                # The cube was carried clear of the blue cube; it must not have slipped low in the jaws.
                phase_ok &= offset[:, 2] > -PLACED_VERTICAL
        valid = finite & within_limits & calm & blue_ok & phase_ok
        self.last_diagnostics = {
            **self.candidate_diagnostics,
            "finite_rate": float(finite.float().mean()),
            "within_limits_rate": float(within_limits.float().mean()),
            "calm_rate": float(calm.float().mean()),
            "blue_ok_rate": float(blue_ok.float().mean()),
            "bilateral_rate": float(bilateral.float().mean()),
            "centered_rate": float(centered.float().mean()),
            "held_rate": float(held.float().mean()),
            "red_flat_rate": float(red_flat.float().mean()),
            "red_height_mean_m": float(red[:, 2].mean()),
            "stack_horizontal_mean_m": float(horizontal_distance(offset).mean()),
            "stack_vertical_mean_m": float(offset[:, 2].mean()),
            "phase_ok_rate": float(phase_ok.float().mean()),
            "valid_rate": float(valid.float().mean()),
        }
        return valid

    # --- phases -------------------------------------------------------------------------------------------------------

    def _approach_rollout(self, fraction: torch.Tensor) -> torch.Tensor:
        """Move the open gripper from home toward the red cube: home to overhead, then down to the pregrasp."""
        home = self._start_batch()
        overhead, pregrasp = self._approach_commands(home)
        first = torch.lerp(home, overhead, (2.0 * fraction).clamp_max(1.0).unsqueeze(-1))
        self._move_robot(first, self.cfg.transition_steps)
        second = torch.lerp(overhead, pregrasp, (2.0 * fraction - 1.0).clamp(0.0, 1.0).unsqueeze(-1))
        target = torch.where((fraction > 0.5).unsqueeze(-1), second, first)
        self._move_robot(target, self.cfg.transition_steps)
        self._simulate(self.cfg.contact_settle_steps // 3, target)
        return target

    def _grasp_rollout(self) -> torch.Tensor:
        """Reach the pregrasp, center the jaws, close on the red cube, and prove the grasp with a small lift."""
        target = self._approach_rollout(torch.ones(self.num_envs, device=self.device))
        target = self._center_open_gripper(target)
        target = target.clone()
        target[:, -1] = GRASP_GRIPPER_POSITION
        self._move_robot(target, self.cfg.grasp_close_steps)
        self._simulate(self.cfg.contact_settle_steps, target)
        proof = torch.zeros((self.num_envs, 3), device=self.device)
        proof[:, 2] = self.cfg.proof_lift
        target = self._carry(target, proof, torch.ones(self.num_envs, device=self.device))
        self._simulate(self.cfg.contact_settle_steps // 3, target)
        return target

    def _restore_check(self, valid: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Restore each row exactly as the reset event will and require it to stay valid."""
        joint_position = _tensor(self.robot.data.joint_pos).detach().clone()
        red, blue = self._cube_pose(self.red).detach().clone(), self._cube_pose(self.blue).detach().clone()
        self._write_robot(joint_position)
        self.robot.set_joint_position_target_index(target=target)
        self._write_cube(self.red, red)
        self._write_cube(self.blue, blue)
        self._commit_reset_state()
        self._simulate(self.cfg.contact_settle_steps, target)
        restored = self._valid(GRASP)
        self.last_diagnostics["connected_valid_count"] = float(valid.sum())
        self.last_diagnostics["restore_valid_count"] = float((valid & restored).sum())
        return valid & restored

    def _candidate_batch(self, phase: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulate one batch of candidates for ``phase`` and return their validity and joint targets."""
        self.candidate_diagnostics = {}
        self.candidate_ik_valid.fill_(True)
        if phase == HOME:
            target = self._start_batch()
            self.candidate_difficulty.zero_()
            return self._valid(HOME), target
        if phase == APPROACH:
            fraction = self._uniform((self.num_envs,), 0.0, 1.0)
            target = self._approach_rollout(fraction)
            self.candidate_difficulty.copy_((APPROACH + fraction) / LAST_PHASE)
            return self._valid(APPROACH), target
        if phase == GRASP:
            target = self._grasp_rollout()
            valid = self._restore_check(self._valid(GRASP), target)
            self.candidate_difficulty.fill_((GRASP + 1.0) / LAST_PHASE)
            self._append_seed_bank("grasp", valid, target)
            return valid, target

        bank = {LIFT: "grasp", TRANSPORT: "lift", PLACE: "transport", RELEASE: "place"}[phase]
        target = self._restore_seed_bank(bank)
        red = self._cube_pose(self.red)
        blue = self._cube_pose(self.blue)
        if phase == RELEASE:
            fraction = self._uniform((self.num_envs,), 0.0, 1.0)
            target = target.clone()
            target[:, -1] = torch.lerp(
                torch.full_like(fraction, GRASP_GRIPPER_POSITION),
                torch.full_like(fraction, RELEASE_GRIPPER_POSITION),
                fraction,
            )
            self._move_robot(target, self.cfg.grasp_close_steps)
            # Give the released cube time to settle on the blue cube.
            self._simulate(2 * self.cfg.contact_settle_steps, target)
            self.candidate_difficulty.copy_((RELEASE + fraction) / LAST_PHASE)
            return self._valid(RELEASE), target

        fraction = self._segment_fraction()
        translation = torch.zeros((self.num_envs, 3), device=self.device)
        if phase == LIFT:
            translation[:, 2] = self.cfg.carry_height - red[:, 2]
        elif phase == TRANSPORT:
            translation[:, :2] = blue[:, :2] - red[:, :2]
        else:
            translation[:, 2] = blue[:, 2] + STACK_HEIGHT + self.cfg.place_height - red[:, 2]
        target = self._carry(target, translation, fraction)
        self._simulate(self.cfg.contact_settle_steps // 3, target)
        valid = self._valid(phase)
        self.candidate_difficulty.copy_((phase + fraction) / LAST_PHASE)
        terminal = valid & (fraction == 1.0)
        if phase == LIFT:
            red_now = self._cube_pose(self.red)
            terminal &= red_now[:, 2] - CUBE_SIZE / 2 - (blue[:, 2] + CUBE_SIZE / 2) >= LIFT_CLEARANCE
        if phase == PLACE:
            offset = stack_offset(self._cube_pose(self.red)[:, :3], self._cube_pose(self.blue)[:, :3])
            terminal &= (horizontal_distance(offset) < PLACED_HORIZONTAL) & (offset[:, 2].abs() <= PLACED_VERTICAL)
            self.last_diagnostics["place_terminal_count"] = float(terminal.sum())
        self._append_seed_bank({LIFT: "lift", TRANSPORT: "transport", PLACE: "place"}[phase], terminal, target)
        return valid, target

    def _flags(self, phase: int, take: torch.Tensor, red_pose: torch.Tensor, blue_pose: torch.Tensor):
        """Milestones the environment restores for accepted rows, measured from the physical state."""
        fixed, moving = jaw_contacts(self.env)
        grasped = (fixed & moving)[take] if phase == GRASP else torch.full_like(take, phase > GRASP, dtype=torch.bool)
        clearance = red_pose[:, 2] - CUBE_SIZE / 2 - (blue_pose[:, 2] + CUBE_SIZE / 2)
        # Phases after the lift have been lifted even when the cube is now low again.
        lifted = grasped & ((clearance >= LIFT_CLEARANCE) | (phase > LIFT))
        return grasped, lifted

    def generate(self) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
        """Generate balanced phases with bounded rejection sampling."""
        rows = {
            name: []
            for name in (
                "joint_position",
                "joint_target",
                "cube_a_pose",
                "cube_b_pose",
                "phase",
                "difficulty",
                "grasped",
                "lifted",
            )
        }
        for phase, phase_name in enumerate(PHASE_NAMES):
            produced_bank = {GRASP: "grasp", LIFT: "lift", TRANSPORT: "transport", PLACE: "place"}.get(phase)

            def branch_ready(name: str | None = produced_bank) -> bool:
                return name is None or (
                    name in self._seed_banks
                    and self._seed_banks[name]["joint_position"].shape[0] >= self.cfg.branch_seed_count
                )

            accepted = attempts = invalid = 0
            while (
                accepted < self.cfg.poses_per_phase or not branch_ready()
            ) and attempts < self.cfg.max_attempts_per_phase:
                valid, joint_target = self._candidate_batch(phase)
                attempts += self.num_envs
                ids = valid.nonzero(as_tuple=False).squeeze(-1)
                invalid += self.num_envs - ids.numel()
                take = ids[: self.cfg.poses_per_phase - accepted]
                bank = self._seed_banks.get(produced_bank or "")
                seed_count = bank["joint_position"].shape[0] if bank else 0
                print(
                    f"[INFO] {phase_name}: batch valid={ids.numel()}/{self.num_envs}, "
                    f"accepted={accepted + take.numel()}, attempted={attempts}, seed_count={seed_count}, "
                    f"diagnostics={self.last_diagnostics}",
                    flush=True,
                )
                if take.numel() == 0:
                    continue
                red_pose = self._cube_pose(self.red)[take].detach().clone()
                blue_pose = self._cube_pose(self.blue)[take].detach().clone()
                grasped, lifted = self._flags(phase, take, red_pose, blue_pose)
                rows["joint_position"].append(_tensor(self.robot.data.joint_pos)[take].detach().clone())
                rows["joint_target"].append(joint_target[take].detach().clone())
                rows["cube_a_pose"].append(red_pose)
                rows["cube_b_pose"].append(blue_pose)
                rows["phase"].append(torch.full((take.numel(),), phase, device=self.device, dtype=torch.long))
                rows["difficulty"].append(self.candidate_difficulty[take].detach().clone().clamp(0.0, 1.0))
                rows["grasped"].append(grasped.detach().clone())
                rows["lifted"].append(lifted.detach().clone())
                accepted += take.numel()
            self.rejections[phase_name] = invalid
            if accepted != self.cfg.poses_per_phase or not branch_ready():
                seed_detail = f"; too few {produced_bank!r} seeds for the next phase" if not branch_ready() else ""
                raise RuntimeError(
                    f"Generated only {accepted}/{self.cfg.poses_per_phase} valid {phase_name!r} poses "
                    f"after {attempts} attempts{seed_detail}. Last batch diagnostics: {self.last_diagnostics}."
                )
            seed_counts = {name: bank["joint_position"].shape[0] for name, bank in self._seed_banks.items()}
            print(
                f"[INFO] {phase_name}: accepted={accepted}, invalid={invalid}, attempted={attempts}, "
                f"seed_banks={seed_counts}",
                flush=True,
            )
        states = {name: torch.cat(parts, dim=0) for name, parts in rows.items()}
        return states, self.rejections


def _parse_args(parser: argparse.ArgumentParser, argv: list[str] | None) -> argparse.Namespace:
    """Parse arguments, also accepting the ``presets=a,b`` form used by the ``isaaclab`` CLI."""
    args, unknown = parser.parse_known_args(argv)
    for token in unknown:
        if not token.startswith("presets="):
            parser.error(f"unrecognized arguments: {token}")
        args.presets = token.removeprefix("presets=")
    return args


def _generator_env_cfg(num_envs: int, device: str, presets: str):
    """The nominal stacking task, driven directly: no dataset resets, rewards, terminations or cube randomization."""
    from isaaclab_tasks.utils.hydra import resolve_presets

    from isaaclab_tutorial.tasks.place_cube.config.so101.env_cfg import SO101CubeEnvCfg

    env_cfg = resolve_presets(SO101CubeEnvCfg(), presets.split(","))
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = device
    for name in ("reset_from_dataset", "cube_a_material", "cube_a_mass", "cube_b_material"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    env_cfg.rewards = None
    env_cfg.terminations = None
    return env_cfg


def generate_main(argv: list[str] | None = None) -> int:
    """CLI entry point for reset generation."""
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description="Generate physics-validated SO-101 cube-stacking reset poses.")
    parser.add_argument("--output", type=Path, default=CUBE_RESET_DATASET)
    parser.add_argument("--poses_per_phase", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_attempts_per_phase", type=int, default=32_768)
    parser.add_argument("--branch_seed_count", type=int, default=32)
    parser.add_argument("--presets", default="newton_mjwarp", help="Comma-separated presets; presets=a,b also works")
    add_launcher_args(parser)
    args = _parse_args(parser, argv)

    from isaaclab.envs import ManagerBasedRLEnv

    cfg = GeneratorCfg(
        poses_per_phase=args.poses_per_phase,
        batch_size=args.batch_size,
        seed=args.seed,
        max_attempts_per_phase=args.max_attempts_per_phase,
        branch_seed_count=args.branch_seed_count,
    )
    env_cfg = _generator_env_cfg(cfg.batch_size, args.device, args.presets)
    env_cfg.seed = cfg.seed
    with launch_simulation(env_cfg, args):
        env = ManagerBasedRLEnv(env_cfg)
        try:
            states, rejections = _Generator(env, cfg).generate()
        finally:
            env.close()
    artifact = save_reset_dataset(
        args.output,
        states,
        generator=asdict(cfg),
        validation={
            "physics": "newton_mjwarp",
            "gravity": True,
            "object_state_writes_after_reset": False,
            "rejections": rejections,
        },
    )
    print(f"[INFO] Wrote {artifact['row_count']} reset poses to {Path(args.output).resolve()}.")
    print(f"[INFO] Content SHA-256: {artifact['content_sha256']}")
    return 0


def view_main(argv: list[str] | None = None) -> int:
    """Cycle a reset artifact in the Newton visualizer without a policy."""
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description="Cycle SO-101 cube-stacking reset poses in the Newton visualizer.")
    parser.add_argument("--dataset", type=Path, default=CUBE_RESET_DATASET)
    parser.add_argument("--steps_per_pose", type=int, default=45)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--phase", type=str, default=None, choices=PHASE_NAMES, help="Only show rows of this phase")
    parser.add_argument("--presets", default="newton_mjwarp", help="Comma-separated presets; presets=a,b also works")
    add_launcher_args(parser)
    args = _parse_args(parser, argv)

    from isaaclab.envs import ManagerBasedRLEnv

    env_cfg = _generator_env_cfg(1, args.device, args.presets)
    with launch_simulation(env_cfg, args):
        env = ManagerBasedRLEnv(env_cfg)
        try:
            states = load_reset_dataset(args.dataset, device=env.device)["states"]
            order = torch.argsort(states["phase"].float() + states["difficulty"] * 0.01, stable=True)
            if args.phase is not None:
                order = order[states["phase"][order] == PHASE_NAMES.index(args.phase)]
            robot = env.scene["robot"]
            zeros = torch.zeros((1, 6), device=env.device)
            for _ in range(args.cycles):
                for row in order.tolist():
                    robot.write_joint_position_to_sim_index(position=states["joint_position"][row : row + 1])
                    robot.write_joint_velocity_to_sim_index(velocity=zeros)
                    for name in ("cube_a", "cube_b"):
                        pose = states[f"{name}_pose"][row : row + 1].clone()
                        pose[:, :3] += env.scene.env_origins[:1]
                        env.scene[name].write_root_pose_to_sim_index(root_pose=pose)
                        env.scene[name].write_root_velocity_to_sim_index(root_velocity=zeros)
                    print(
                        f"[RESET] row={row} phase={PHASE_NAMES[int(states['phase'][row])]} "
                        f"difficulty={float(states['difficulty'][row]):.3f} "
                        f"grasped={bool(states['grasped'][row])} lifted={bool(states['lifted'][row])}",
                        flush=True,
                    )
                    for _ in range(args.steps_per_pose):
                        if not env.sim.is_running():
                            return 0
                        robot.set_joint_position_target_index(target=states["joint_target"][row : row + 1])
                        env.scene.write_data_to_sim()
                        env.sim.step()
                        env.scene.update(env.physics_dt)
        finally:
            env.close()
    return 0
