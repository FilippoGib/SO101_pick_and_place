"""Generate and inspect simulator-validated cube/cylinder reset states."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from isaaclab_tutorial.assets import CUBE_RESET_DATASET
from isaaclab_tutorial.tasks.place_cube.config.so101.env_cfg import (
    PREGRASP_GRIPPER_POSITION,
    TABLETOP_CUBE_POSITION,
    WORKSHOP_INITIAL_JOINT_POSITION,
)
from isaaclab_tutorial.tasks.place_cube.reset.dataset import load_reset_dataset, save_reset_dataset
from isaaclab_tutorial.tasks.place_vial.reset.generator import (
    WORKSHOP_PREGRASP_JOINT_POSITION,
    WORKSHOP_TASK_WAYPOINTS,
)


@dataclass(frozen=True)
class GeneratorCfg:
    poses_per_phase: int = 128
    seed: int = 42
    validation_steps: int = 30

    def __post_init__(self) -> None:
        if self.poses_per_phase <= 0:
            raise ValueError("poses_per_phase must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.validation_steps <= 0:
            raise ValueError("validation_steps must be positive")


def _yaw(angle: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            torch.zeros_like(angle),
            torch.zeros_like(angle),
            torch.sin(angle / 2),
            torch.cos(angle / 2),
        ),
        dim=-1,
    )


def generate_states(cfg: GeneratorCfg) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    count = len(range(8)) * cfg.poses_per_phase
    phases = torch.arange(8, dtype=torch.long).repeat_interleave(cfg.poses_per_phase)
    noise = torch.rand((count, 2), generator=generator) - 0.5
    phase_joints = torch.tensor(WORKSHOP_INITIAL_JOINT_POSITION, dtype=torch.float32).repeat(8, 1)
    phase_joints[1] = torch.tensor((*WORKSHOP_PREGRASP_JOINT_POSITION[:5], PREGRASP_GRIPPER_POSITION))
    phase_joints[2] = torch.tensor(WORKSHOP_TASK_WAYPOINTS[3][0])
    phase_joints[3] = torch.tensor(WORKSHOP_TASK_WAYPOINTS[3][1])
    phase_joints[4] = torch.tensor(WORKSHOP_TASK_WAYPOINTS[4][0])
    phase_joints[5] = torch.tensor(WORKSHOP_TASK_WAYPOINTS[5][1])
    phase_joints[6] = torch.tensor(WORKSHOP_TASK_WAYPOINTS[6][1])
    phase_joints[7] = torch.tensor(WORKSHOP_TASK_WAYPOINTS[7][2])
    joint_position = phase_joints[phases].clone()
    joint_target = joint_position.clone()
    cube_pose = torch.zeros(count, 7)
    cylinder_pose = torch.zeros(count, 7)
    cube_pose[:, :3] = torch.tensor(TABLETOP_CUBE_POSITION)
    cylinder_pose[:, :3] = torch.tensor((0.18, 0.08, 0.035))
    cube_pose[:, 3:] = torch.tensor((0.0, 0.7071068, 0.0, 0.7071068))
    cylinder_pose[:, 6] = 1.0

    home = phases == 0
    cube_pose[home, :2] += noise[home] * torch.tensor((0.06, 0.08))
    cylinder_pose[home, :2] += noise[home] * torch.tensor((0.01, 0.01))
    cylinder_pose[home, 3:] = _yaw(noise[:, 0] * 0.2)[home]

    grasped = phases >= 2
    lifted = phases >= 3
    cube_pose[grasped, 2] = 0.12
    cube_pose[phases >= 5, :3] = torch.tensor((0.18, 0.08, 0.10))
    # Leave a small clearance above the cylinder top for the first validated step;
    # the cube settles onto the support without starting in mesh penetration.
    cube_pose[phases >= 6, 2] = 0.065
    joint_position[:, -1] = torch.where(
        phases == 0,
        torch.tensor(PREGRASP_GRIPPER_POSITION),
        joint_position[:, -1],
    )
    joint_target.copy_(joint_position)
    return {
        "joint_position": joint_position,
        "joint_target": joint_target,
        "cube_pose": cube_pose,
        "cylinder_pose": cylinder_pose,
        "phase": phases,
        "difficulty": phases.float() / 7.0,
        "grasped": grasped,
        "lifted": lifted,
    }


def _parse_args(parser: argparse.ArgumentParser, argv: list[str] | None) -> argparse.Namespace:
    args, unknown = parser.parse_known_args(argv)
    for token in unknown:
        if not token.startswith("presets="):
            parser.error(f"unrecognized arguments: {token}")
    return args


def _generator_env_cfg(num_envs: int, device: str, presets: str):
    from isaaclab_tasks.utils.hydra import resolve_presets

    from isaaclab_tutorial.tasks.place_cube.config.so101.env_cfg import SO101CubeEnvCfg

    env_cfg = resolve_presets(SO101CubeEnvCfg(), presets.split(","))
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = device
    env_cfg.events.reset_from_dataset = None
    env_cfg.events.cube_material = None
    env_cfg.events.cube_mass = None
    env_cfg.rewards = None
    env_cfg.terminations = None
    return env_cfg


def validate_states_in_sim(
    states: dict[str, torch.Tensor], *, steps: int, device: str, presets: str, launcher_args
) -> dict[str, float]:
    """Run every generated reset row in the cube scene and reject unstable states."""
    from isaaclab.app import launch_simulation
    from isaaclab.envs import ManagerBasedRLEnv

    env_cfg = _generator_env_cfg(len(states["phase"]), device, presets)
    with launch_simulation(env_cfg, launcher_args):
        env = ManagerBasedRLEnv(env_cfg)
        try:
            ids = torch.arange(env.num_envs, device=env.device)
            robot = env.scene["robot"]
            zeros = torch.zeros((env.num_envs, 6), device=env.device)
            for name in ("joint_position", "joint_target", "cube_pose", "cylinder_pose"):
                states[name] = states[name].to(env.device)
            robot.write_joint_position_to_sim_index(position=states["joint_position"], env_ids=ids)
            robot.write_joint_velocity_to_sim_index(velocity=zeros, env_ids=ids)
            robot.set_joint_position_target_index(target=states["joint_target"], env_ids=ids)
            robot.set_joint_velocity_target_index(target=zeros, env_ids=ids)
            for name in ("cube", "cylinder"):
                pose = states[f"{name}_pose"].clone()
                pose[:, :3] += env.scene.env_origins
                env.scene[name].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
                env.scene[name].write_root_velocity_to_sim_index(
                    root_velocity=torch.zeros_like(zeros), env_ids=ids
                )
            for _ in range(steps):
                env.scene.write_data_to_sim()
                env.sim.step()
                env.scene.update(env.physics_dt)
            cube = env.scene["cube"]
            cylinder = env.scene["cylinder"]
            finite = torch.isfinite(cube.data.root_pos_w.torch).all() & torch.isfinite(
                cylinder.data.root_pos_w.torch
            ).all()
            if not bool(finite.all()):
                raise RuntimeError("Cube reset validation produced non-finite object poses.")
            speed = torch.linalg.vector_norm(cube.data.root_lin_vel_w.torch, dim=-1)
            max_speed = float(speed.max().item())
            if max_speed > 2.0:
                phase_max = [
                    float(speed[states["phase"].to(env.device) == phase].max().item())
                    for phase in range(8)
                ]
                raise RuntimeError(
                    f"Cube reset validation produced an unstable cube (max speed {max_speed:.3f} m/s; "
                    f"per-phase {phase_max})."
                )
            return {"validated_rows": float(env.num_envs), "max_cube_speed_mps": max_speed}
        finally:
            env.close()


def generate_main(argv: list[str] | None = None) -> int:
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description="Generate cube/cylinder SO-101 reset poses.")
    parser.add_argument("--output", type=Path, default=CUBE_RESET_DATASET)
    parser.add_argument("--poses_per_phase", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_steps", type=int, default=30)
    parser.add_argument("--presets", default="newton_mjwarp")
    add_launcher_args(parser)
    args = _parse_args(parser, argv)
    cfg = GeneratorCfg(
        poses_per_phase=args.poses_per_phase,
        seed=args.seed,
        validation_steps=args.validation_steps,
    )
    states = generate_states(cfg)
    validation = validate_states_in_sim(
        states,
        steps=cfg.validation_steps,
        device=args.device,
        presets=args.presets,
        launcher_args=args,
    )
    artifact = save_reset_dataset(
        args.output,
        states,
        generator=asdict(cfg),
        validation={
            "physics": "newton_mjwarp",
            "gravity": True,
            "trajectory_validation": validation,
        },
    )
    print(f"[INFO] Wrote {artifact['row_count']} cube/cylinder reset poses to {Path(args.output).resolve()}.")
    print(f"[INFO] Content SHA-256: {artifact['content_sha256']}")
    return 0


def view_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect cube/cylinder reset dataset metadata.")
    parser.add_argument("--dataset", type=Path, default=CUBE_RESET_DATASET)
    args = _parse_args(parser, argv)
    artifact = load_reset_dataset(args.dataset)
    counts = torch.bincount(artifact["states"]["phase"], minlength=8).tolist()
    print(f"[INFO] Dataset: {Path(args.dataset).resolve()}")
    print(f"[INFO] Rows by phase: {counts}")
    print(f"[INFO] Content SHA-256: {artifact['content_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(generate_main())
