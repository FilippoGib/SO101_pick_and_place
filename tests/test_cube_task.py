"""Kit-less contracts for the cube-on-cylinder task."""

import gymnasium as gym
import torch

from isaaclab_tasks.utils.hydra import resolve_presets

import isaaclab_tutorial.tasks  # noqa: F401
from isaaclab_tutorial.assets import CUBE_RESET_DATASET, CUBE_USD, CYLINDER_USD
from isaaclab_tutorial.tasks.place_cube.config.so101.camera_env_cfg import SO101CubeCameraEnvCfg
from isaaclab_tutorial.tasks.place_cube.config.so101.env_cfg import SO101CubeEnvCfg
from isaaclab_tutorial.tasks.place_cube.mdp import rewards, terminations
from isaaclab_tutorial.tasks.place_cube.mdp.geometry import cube_on_cylinder
from isaaclab_tutorial.tasks.place_cube.mdp.progress import PlacementProgress
from isaaclab_tutorial.tasks.place_cube.reset.curriculum import ALL_PHASES
from isaaclab_tutorial.tasks.place_cube.reset.dataset import load_reset_dataset
from isaaclab_tutorial.tasks.place_cube.reset.generator import GeneratorCfg, generate_states


def test_cube_assets_and_task_registration():
    assert CUBE_USD.is_file()
    assert CYLINDER_USD.is_file()
    assert "IsaacTutorial-Place-Cube-SO101" in gym.registry
    assert "IsaacTutorial-Place-Cube-SO101-Camera" in gym.registry


def test_cube_reset_schema_contains_cube_and_cylinder_poses():
    artifact = load_reset_dataset(CUBE_RESET_DATASET)
    assert artifact["format"] == "so101-cube-cylinder-reset-dataset"
    assert tuple(artifact["states"]["cube_pose"].shape[1:]) == (7,)
    assert tuple(artifact["states"]["cylinder_pose"].shape[1:]) == (7,)
    assert artifact["validation"]["physics"] == "newton_mjwarp"
    assert artifact["validation"]["trajectory_validation"]["validated_rows"] == 1024.0


def test_cube_reset_generator_produces_all_phases():
    states = generate_states(GeneratorCfg(poses_per_phase=2))
    assert states["phase"].tolist() == [phase for phase in range(8) for _ in range(2)]
    assert states["cube_pose"].shape == (16, 7)
    assert states["cylinder_pose"].shape == (16, 7)
    assert not torch.equal(states["joint_position"][0], states["joint_position"][3])


def test_cube_training_resets_sample_all_phases():
    cfg = resolve_presets(SO101CubeEnvCfg(), ["newton_mjwarp"])
    assert cfg.events.reset_from_dataset.params["phase_weights"] == ALL_PHASES


def test_cube_rewards_remain_finite_for_nonfinite_physics_state(monkeypatch):
    progress = PlacementProgress(2, "cpu")
    progress.grasped[:] = True
    progress.inside[:] = False
    monkeypatch.setattr(rewards, "placement_progress", lambda env: progress)
    monkeypatch.setattr(rewards, "cube_on_cylinder", lambda env: torch.tensor([float("nan"), 0.01]))
    result = rewards.held_goal(object())
    assert torch.isfinite(result).all()
    assert result.tolist()[0] == 0.0


def test_cube_drop_termination_uses_cube_rest_height(monkeypatch):
    class TensorWrapper:
        def __init__(self, tensor):
            self.torch = tensor

    class Cube:
        data = type(
            "Data", (), {"root_pos_w": TensorWrapper(torch.tensor([[0.0, 0.0, 0.065], [0.0, 0.0, 0.09]]))}
        )()

    class Scene:
        env_origins = torch.zeros(2, 3)

        def __getitem__(self, name):
            return {"cube": Cube()}[name]

    class Env:
        scene = Scene()

    progress = PlacementProgress(2, "cpu")
    progress.lifted[:] = True
    monkeypatch.setattr(terminations, "placement_progress", lambda env: progress)
    monkeypatch.setattr(terminations, "cube_supported", lambda env: torch.tensor([False, False]))
    result = terminations.cube_dropped(Env())
    assert result.tolist() == [True, False]


def test_cube_is_exactly_two_centimetres():
    assert "float3 xformOp:scale = (0.02, 0.02, 0.02)" in CUBE_USD.read_text()


def test_cylinder_is_solid_and_seated_on_the_table():
    cylinder_text = CYLINDER_USD.read_text()
    assert 'def Mesh "Visual"' in cylinder_text
    assert 'apiSchemas = ["PhysicsCollisionAPI"]' in cylinder_text
    assert "(0.03,0,0)" in cylinder_text
    assert "(0.03,0,0.01)" in cylinder_text
    cfg = resolve_presets(SO101CubeEnvCfg(), ["newton_mjwarp"])
    assert cfg.scene.cylinder.init_state.pos[2] == 0.035


def test_cube_on_cylinder_requires_centered_top_support():
    cube_pos = torch.tensor([[0.0, 0.0, 0.055], [0.030, 0.0, 0.055]])
    cube_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    cylinder_pos = torch.zeros(2, 3)
    assert cube_on_cylinder(cube_pos, cube_quat, cylinder_pos, 0.045, 0.03).tolist() == [True, False]


def test_cube_state_and_camera_configs_share_actions_and_physics():
    state = resolve_presets(SO101CubeEnvCfg(), ["newton_mjwarp"])
    camera = resolve_presets(SO101CubeCameraEnvCfg(), ["newton_mjwarp", "newton_renderer"])
    assert camera.actions.to_dict() == state.actions.to_dict()
    assert camera.sim.dt == state.sim.dt
    assert camera.scene.cube.spawn.usd_path == str(CUBE_USD)
    assert state.sim.physics.solver_cfg.njmax == 512
    assert state.sim.physics.solver_cfg.nconmax == 512
