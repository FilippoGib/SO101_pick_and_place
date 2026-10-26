"""Kit-less contracts for the cube-in-cup task."""

import math

import gymnasium as gym
import torch

from isaaclab_tasks.utils.hydra import resolve_presets

import isaaclab_tutorial.tasks  # noqa: F401
from isaaclab_tutorial.assets import CUBE_USD, CUP_USD
from isaaclab_tutorial.tasks.place_cube.config.so101.camera_env_cfg import SO101CubeCameraEnvCfg
from isaaclab_tutorial.tasks.place_cube.config.so101.env_cfg import SO101CubeEnvCfg
from isaaclab_tutorial.tasks.place_cube.mdp.geometry import cube_fully_inside_cup, cup_upright


def test_cube_assets_and_task_registration():
    assert CUBE_USD.is_file()
    assert CUP_USD.is_file()
    assert "IsaacTutorial-Place-Cube-SO101" in gym.registry
    assert "IsaacTutorial-Place-Cube-SO101-Camera" in gym.registry


def test_cube_is_exactly_two_centimetres():
    assert "float3 xformOp:scale = (0.02, 0.02, 0.02)" in CUBE_USD.read_text()


def test_cup_is_hollow_tapered_and_seated_on_the_table():
    cup_text = CUP_USD.read_text()
    assert 'def Mesh "Visual"' in cup_text
    assert 'def Mesh "Collision"' in cup_text
    assert "(0.050000, 0.000000, 0.080000)" in cup_text
    assert "(0.040000, 0.000000, 0.000000)" in cup_text
    cfg = resolve_presets(SO101CubeEnvCfg(), ["newton_mjwarp"])
    assert cfg.scene.cup.init_state.pos[2] == 0.035


def test_cube_containment_requires_all_corners_inside_and_cup_upright():
    cube_pos = torch.tensor([[0.0, 0.0, 0.044], [0.030, 0.0, 0.044]])
    cube_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    cup_pos = torch.zeros(2, 3)
    cup_quat = cube_quat.clone()
    assert cube_fully_inside_cup(cube_pos, cube_quat, cup_pos, cup_quat).tolist() == [True, False]
    assert cup_upright(cup_quat).tolist() == [True, True]
    tilted = torch.tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]])
    assert not cup_upright(tilted).item()


def test_cube_state_and_camera_configs_share_actions_and_physics():
    state = resolve_presets(SO101CubeEnvCfg(), ["newton_mjwarp"])
    camera = resolve_presets(SO101CubeCameraEnvCfg(), ["newton_mjwarp", "newton_renderer"])
    assert camera.actions.to_dict() == state.actions.to_dict()
    assert camera.sim.dt == state.sim.dt
    assert camera.scene.cube.spawn.usd_path == str(CUBE_USD)
