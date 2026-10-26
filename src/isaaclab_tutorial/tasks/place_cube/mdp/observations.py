"""Cube/cup task observations and deployable wrist-camera observations."""

from collections.abc import Sequence

import torch

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.images import CameraFrameStack

from .progress import placement_progress
from .task import cube_in_cup, cube_upright, jaw_contacts


def joint_target(env, asset_cfg):
    return env.scene[asset_cfg.name].data.joint_pos_target.torch[:, asset_cfg.joint_ids]


def encoder_joint_pos(env, asset_cfg):
    return env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids] + getattr(
        env, "_so101_encoder_bias", 0.0
    )


def encoder_joint_target(env, asset_cfg):
    return joint_target(env, asset_cfg) + getattr(env, "_so101_encoder_bias", 0.0)


def cube_in_cup_frame(env):
    return cube_in_cup(env)[0]


def placement_features(env):
    position, _, lowest = cube_in_cup(env)
    cube = env.scene["cube"]
    speed = torch.linalg.vector_norm(cube.data.root_lin_vel_w.torch, dim=-1)
    return torch.stack((position[:, 0], position[:, 1], lowest, cube_upright(env), speed), dim=-1)


def progress_flags(env):
    progress = placement_progress(env)
    return torch.stack((progress.grasped, progress.lifted, progress.inside), dim=-1).float()


def jaw_contact_flags(env):
    fixed, moving = jaw_contacts(env)
    return torch.stack((fixed, moving, fixed & moving), dim=-1).float()


class wrist_rgb(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env):
        super().__init__(cfg, env)
        params = cfg.params
        self.frames = CameraFrameStack(env.num_envs, env.device, params.get("history_length", 1), channel_first=True)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None):
        self.frames.reset(env_ids)

    def __call__(
        self,
        env,
        sensor_cfg: SceneEntityCfg,
        projection_size=(48, 64),
        focal_length_pixels=64.0,
        exposure_range=(1.0, 1.0),
        contrast_range=(1.0, 1.0),
        white_balance_range=(1.0, 1.0),
        brightness_range=(0.0, 0.0),
        gamma_range=(1.0, 1.0),
        blur_range=(0.0, 0.0),
        focal_scale_range=(1.0, 1.0),
        principal_point_pixels=0.0,
        radial_distortion_range=(0.0, 0.0),
        history_length=1,
    ):
        image = env.scene.sensors[sensor_cfg.name].data.output["rgb"].torch[..., :3]
        image = image.permute(0, 3, 1, 2).float() / 255.0
        return self.frames(image.permute(0, 2, 3, 1))
