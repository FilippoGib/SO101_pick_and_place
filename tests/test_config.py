"""Configuration contracts for the state and camera tasks, with and without the ``sim2real`` preset."""

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab_tasks.utils.hydra import resolve_presets

import isaaclab_tutorial.tasks  # noqa: F401  (registers the tasks)
from isaaclab_tutorial.assets import VIAL_LABELED_USD, VIAL_USD
from isaaclab_tutorial.tasks.place_vial.config.so101.agents.rsl_rl_distillation_cfg import (
    BoundedTeacherDistillation,
    SO101CameraDistillationRunnerCfg,
)
from isaaclab_tutorial.tasks.place_vial.config.so101.agents.rsl_rl_ppo_cfg import SO101StatePPORunnerCfg
from isaaclab_tutorial.tasks.place_vial.config.so101.camera_env_cfg import (
    CameraSim2RealEventsCfg,
    SO101VialCameraEnvCfg,
)
from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import (
    BASE_HEIGHT_CORRECTION,
    PREGRASP_GRIPPER_POSITION,
    EventsCfg,
    Sim2RealEventsCfg,
    SO101VialEnvCfg,
    _remove_non_gripper_colliders,
)
from isaaclab_tutorial.tasks.place_vial.reset.curriculum import ALL_PHASES, CANONICAL_START
from isaaclab_tutorial.utils import evaluation


def _state(*presets):
    return resolve_presets(SO101VialEnvCfg(), ["newton_mjwarp", *presets])


def _camera(*presets):
    return resolve_presets(SO101VialCameraEnvCfg(), ["newton_mjwarp", "newton_renderer", *presets])


def test_tasks_are_registered_with_rsl_rl_entry_points():
    ids = sorted(task for task in gym.registry if task.startswith("IsaacTutorial-"))
    assert ids == [
        "IsaacTutorial-Inspect-Cube-SO101",
        "IsaacTutorial-Inspect-SO101",
        "IsaacTutorial-Place-Cube-SO101",
        "IsaacTutorial-Place-Vial-SO101",
        "IsaacTutorial-Place-Vial-SO101-Camera",
    ]
    cube_inspection, inspection, cube, state, camera = (gym.spec(task).kwargs for task in ids)
    assert cube["env_cfg_entry_point"].endswith("place_cube.config.so101.env_cfg:SO101CubeEnvCfg")
    assert cube["rsl_rl_cfg_entry_point"].endswith("agents.rsl_rl_ppo_cfg:SO101CubeStatePPORunnerCfg")
    assert cube_inspection["env_cfg_entry_point"].endswith("inspection_env_cfg:SO101CubeInspectionEnvCfg")
    assert "rsl_rl_cfg_entry_point" not in cube_inspection
    assert inspection["env_cfg_entry_point"].endswith("inspection_env_cfg:SO101InspectionEnvCfg")
    assert "rsl_rl_cfg_entry_point" not in inspection
    assert state["env_cfg_entry_point"].endswith("env_cfg:SO101VialEnvCfg")
    assert state["rsl_rl_cfg_entry_point"].endswith("rsl_rl_ppo_cfg:SO101StatePPORunnerCfg")
    assert camera["env_cfg_entry_point"].endswith("camera_env_cfg:SO101VialCameraEnvCfg")
    assert camera["rsl_rl_cfg_entry_point"].endswith("rsl_rl_distillation_cfg:SO101CameraDistillationRunnerCfg")


def test_nominal_state_task():
    cfg = _state()
    assert cfg.scene.num_envs == 4096
    assert cfg.decimation == 4
    assert cfg.sim.dt == pytest.approx(1.0 / 120.0)
    assert cfg.episode_length_s == 20.0
    assert type(cfg.events) is EventsCfg
    assert cfg.events.vial_material.mode == "startup"
    assert cfg.events.reset_from_dataset.params["phase_weights"] == ALL_PHASES
    assert cfg.actions.arm_action.scale == pytest.approx(0.033)
    assert cfg.actions.gripper_action.scale == pytest.approx(0.02)
    assert cfg.scene.robot.init_state.pos == (-0.05, 0.0, 0.0)
    assert cfg.scene.robot.init_state.joint_pos["gripper"] == pytest.approx(PREGRASP_GRIPPER_POSITION)
    assert cfg.scene.robot.spawn.shoulder_response is None
    assert cfg.scene.robot.spawn.hide_camera_housing is False
    assert cfg.scene.vial.spawn.usd_path == str(VIAL_USD)
    assert cfg.scene.vial.spawn.randomize_diameters is False
    for name in ("support", "fixed_jaw_support", "moving_jaw_support"):
        assert getattr(cfg.scene, name) is None
    assert cfg.rewards.jaw_support_force is None
    assert cfg.rewards.approach_progress.params["height_offset"] == 0.0
    assert cfg.rewards.joint_limits.weight < 0


def test_sim2real_state_task():
    cfg = _state("sim2real")
    assert cfg.episode_length_s == 30.0
    assert type(cfg.events) is Sim2RealEventsCfg
    assert cfg.events.vial_material.mode == "reset"
    assert cfg.events.vial_mass.mode == "reset"
    # The dataset's held rows were generated with the base 5 mm lower, so sim2real trains from home only.
    assert cfg.events.reset_from_dataset.params["phase_weights"] == CANONICAL_START
    assert cfg.actions.arm_action.scale["shoulder_lift"] == pytest.approx(0.12)
    assert cfg.actions.arm_action.scale["elbow_flex"] == pytest.approx(0.07)
    assert cfg.scene.robot.init_state.pos == (-0.05, 0.0, BASE_HEIGHT_CORRECTION)
    assert cfg.scene.robot.spawn.shoulder_response == (0.22, 0.1, 0.4)
    assert cfg.scene.vial.spawn.randomize_diameters is True
    assert cfg.scene.support is not None
    assert cfg.scene.fixed_jaw_support.filter_prim_paths_expr == ["{ENV_REGEX_NS}/Desk", "{ENV_REGEX_NS}/Support"]
    assert cfg.rewards.jaw_support_force is not None
    assert cfg.rewards.approach_progress.params["height_offset"] == 0.008


def test_camera_task_inherits_state_task_and_adds_the_wrist_camera():
    nominal = _camera()
    assert nominal.scene.num_envs == 1024
    assert type(nominal.events) is EventsCfg
    assert nominal.scene.robot.spawn.hide_camera_housing is False
    assert nominal.scene.vial.spawn.usd_path == str(VIAL_LABELED_USD)
    assert (nominal.scene.wrist_camera.width, nominal.scene.wrist_camera.height) == (80, 60)
    image = nominal.observations.wrist_rgb.image.params
    assert image["projection_size"] == (48, 64)
    assert image["history_length"] == 2
    assert image["focal_scale_range"] == (1.0, 1.0)
    assert image["blur_range"] == (0.0, 0.0)
    # The policy-resolution focal length is the nominal 64 px wrist camera.
    assert image["focal_length_pixels"] == pytest.approx(64 * 13.6 / 20.955)

    randomized = _camera("sim2real")
    assert type(randomized.events) is CameraSim2RealEventsCfg
    assert randomized.events.camera_mount is not None
    image = randomized.observations.wrist_rgb.image.params
    assert image["focal_scale_range"] == (0.95, 1.05)
    assert image["radial_distortion_range"] == (-0.04, 0.04)
    # The teacher and student see the same physics.
    state = _state("sim2real")
    assert randomized.scene.robot.init_state.pos == state.scene.robot.init_state.pos
    assert randomized.actions.arm_action.scale == state.actions.arm_action.scale
    for name in vars(state.events):
        assert getattr(randomized.events, name).to_dict() == getattr(state.events, name).to_dict()


def test_agent_observation_groups_exist_in_their_tasks():
    state, camera = _state(), _camera()
    for groups in SO101StatePPORunnerCfg().obs_groups.values():
        assert all(hasattr(state.observations, group) for group in groups)
    for groups in SO101CameraDistillationRunnerCfg().obs_groups.values():
        assert all(hasattr(camera.observations, group) for group in groups)
    # The teacher input in the camera task is exactly the state actor input.
    assert camera.observations.teacher_state.to_dict() == state.observations.policy.to_dict()


@pytest.mark.parametrize("presets", [(), ("sim2real",)])
def test_play_uses_home_starts_and_pins_vial_only_without_sim2real(monkeypatch, presets):
    monkeypatch.setattr(evaluation, "EXACT_EVALUATION_ACTIVE", True)
    cfg = _state(*presets)
    cfg.scene.num_envs = 1024
    material = cfg.events.vial_material.params["static_friction_range"]
    cfg.play_mode()
    assert cfg.scene.num_envs == 1024
    assert cfg.events.reset_from_dataset.params["sequential"] is True
    assert cfg.events.reset_from_dataset.params["phase_weights"] == CANONICAL_START
    expected = material if presets else (0.7, 0.7)
    assert cfg.events.vial_material.params["static_friction_range"] == expected


def test_reset_generator_uses_the_nominal_state_scene():
    from isaaclab_tutorial.tasks.place_vial.reset.generator import _generator_env_cfg

    cfg = _generator_env_cfg(8, "cpu", "newton_mjwarp")
    assert cfg.scene.num_envs == 8
    assert cfg.events.reset_from_dataset is None
    assert cfg.events.vial_mass.params["mass_distribution_params"] == (0.02, 0.02)
    assert cfg.rewards is None and cfg.terminations is None


def test_only_jaw_and_camera_mount_colliders_remain():
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    for link in ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"):
        body = UsdGeom.Xform.Define(stage, f"/Robot/{link}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        mesh = UsdGeom.Mesh.Define(stage, f"/Robot/{link}/collisions/mesh").GetPrim()
        UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr("convexHull")
    camera = UsdGeom.Cube.Define(stage, "/Robot/gripper/collisions/camera_mount/Cube").GetPrim()
    UsdPhysics.CollisionAPI.Apply(camera)
    _remove_non_gripper_colliders(root)
    kept = {str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)}
    assert kept == {
        "/Robot/gripper/collisions/mesh",
        "/Robot/moving_jaw_so101_v1/collisions/mesh",
        "/Robot/gripper/collisions/camera_mount/Cube",
    }


def test_rolling_resistance_only_enables_support_contacts():
    from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import configure_support_rolling_contacts

    builder = SimpleNamespace(
        shape_label=[
            "/env_0/Support/geometry",
            "/env_0/Desk/collision",
            "/env_0/Vial/body_collider",
            "/env_0/Robot/gripper/collision",
            "/env_0/Rack/collision",
            "/env_1/Support/geometry",
        ],
        shape_world=[0, 0, 0, 0, 0, 1],
        shape_material_mu_rolling=[0.05] * 6,
        shape_material_mu_torsional=[0.005] * 6,
        custom_attributes={"mujoco:condim": SimpleNamespace(values={i: 3 for i in range(6)})},
    )
    configure_support_rolling_contacts(builder, (0.0002, 0.002), (0.001, 0.005))
    assert list(builder.custom_attributes["mujoco:condim"].values.values()) == [6, 6, 3, 3, 3, 6]
    assert 0.0002 <= builder.shape_material_mu_rolling[0] <= 0.002
    assert builder.shape_material_mu_rolling[2:5] == [0.0, 0.0, 0.0]


def test_distillation_labels_are_clamped_to_the_executed_action():
    algorithm = object.__new__(BoundedTeacherDistillation)
    algorithm.student = lambda obs, stochastic_output: torch.full((obs.shape[0], 2), 0.25)
    algorithm.teacher = lambda obs: torch.tensor([[4.0, -3.0]]).expand(obs.shape[0], -1)
    algorithm.transition = SimpleNamespace()
    actions = algorithm.act(torch.zeros(3, 1))
    assert torch.equal(actions, torch.full((3, 2), 0.25))
    assert torch.equal(algorithm.transition.privileged_actions, torch.tensor([[1.0, -1.0]]).expand(3, -1))


def _cube_inspection(*presets):
    from isaaclab_tutorial.tasks.place_cube.config.so101.inspection_env_cfg import SO101CubeInspectionEnvCfg

    return resolve_presets(SO101CubeInspectionEnvCfg(), ["newton_mjwarp", *presets])


def test_cube_scene_replaces_the_vial_and_rack_with_two_resting_cubes():
    from isaaclab_tutorial.assets import CUBE_USD
    from isaaclab_tutorial.tasks.place_cube.config.so101.env_cfg import CUBE_SIZE, TABLETOP_Z

    cfg = _cube_inspection()
    scene = cfg.scene
    assert not hasattr(scene, "vial") and not hasattr(scene, "rack")
    assert scene.cube_a.init_state.pos[:2] == pytest.approx((0.231, -0.017))  # the vial's tabletop position
    assert scene.cube_b.init_state.pos[:2] == pytest.approx((0.2101682871, 0.1101424648))  # rack footprint center
    for cube, color in ((scene.cube_a, "red"), (scene.cube_b, "blue")):
        assert cube.spawn.usd_path == str(CUBE_USD)
        assert cube.spawn.variants == {"Color": color}
        assert cube.init_state.pos[2] - CUBE_SIZE / 2 == pytest.approx(TABLETOP_Z, abs=0.001)
    # Grasp detection sees only the red cube; red/blue contact has its own sensor.
    assert scene.moving_jaw_contact.filter_prim_paths_expr == ["{ENV_REGEX_NS}/CubeA"]
    assert scene.cube_stack_contact.prim_path == "{ENV_REGEX_NS}/CubeA"
    assert scene.cube_stack_contact.filter_prim_paths_expr == ["{ENV_REGEX_NS}/CubeB"]
    assert scene.support is None and scene.fixed_jaw_support is None
    assert scene.robot.init_state.pos == (-0.05, 0.0, 0.0)
    assert cfg.rewards is None and cfg.terminations is None


def test_cube_scene_sim2real_mirrors_the_vial_scene():
    scene = _cube_inspection("sim2real").scene
    vial_scene = _state("sim2real").scene
    assert scene.robot.init_state.pos == vial_scene.robot.init_state.pos == (-0.05, 0.0, BASE_HEIGHT_CORRECTION)
    assert scene.robot.spawn.shoulder_response == vial_scene.robot.spawn.shoulder_response
    assert scene.robot.spawn.elbow_response == vial_scene.robot.spawn.elbow_response
    assert scene.support.spawn.size == vial_scene.support.spawn.size
    assert scene.support.init_state.pos == vial_scene.support.init_state.pos
    assert scene.fixed_jaw_support.prim_path == vial_scene.fixed_jaw_support.prim_path
    assert scene.moving_jaw_support.prim_path == vial_scene.moving_jaw_support.prim_path


def _cube(*presets):
    from isaaclab_tutorial.tasks.place_cube.config.so101.env_cfg import SO101CubeEnvCfg

    return resolve_presets(SO101CubeEnvCfg(), ["newton_mjwarp", *presets])


@pytest.mark.parametrize("presets", [(), ("sim2real",)])
def test_cube_task_reuses_the_vial_actions_and_timing(presets):
    cfg, vial = _cube(*presets), _state(*presets)
    for name in ("arm_action", "gripper_action"):
        cube_term, vial_term = getattr(cfg.actions, name), getattr(vial.actions, name)
        assert cube_term.joint_names == vial_term.joint_names
        assert cube_term.scale == vial_term.scale
        assert cube_term.use_zero_offset and type(cube_term) is type(vial_term)
    assert (cfg.decimation, cfg.sim.dt, cfg.episode_length_s) == (vial.decimation, vial.sim.dt, vial.episode_length_s)
    assert cfg.episode_length_s == (30.0 if presets else 20.0)
    assert (cfg.scene.support is None) == (not presets)


def test_cube_teacher_observations_mirror_the_vial_teacher():
    cfg, vial = _cube(), _state()
    policy, critic = cfg.observations.policy, cfg.observations.critic
    for name in ("cube_a", "cube_b"):
        term = getattr(policy, name)
        assert term.params["body_asset_cfg"].name == name
        assert term.params["base_asset_cfg"].name == "robot"
    shared = ("joint_pos", "joint_vel", "joint_target", "previous_action", "end_effector", "progress")
    for name in shared:
        # The cube package has its own copies of the task terms, so compare by name.
        assert getattr(policy, name).func.__name__ == getattr(vial.observations.policy, name).func.__name__
    assert set(vars(critic)) == set(vars(policy)) | {"contact"}
    assert not policy.enable_corruption and policy.concatenate_terms


def test_cube_teacher_terminations_and_rewards():
    cfg = _cube()
    terms = cfg.terminations
    for name in ("cube_a", "cube_b"):
        bounds = getattr(terms, f"{name}_out_of_bounds").params
        assert bounds["asset_cfg"].name == name
        assert bounds["in_bound_range"]["z"][0] < 0.035 - 0.015  # below the desk top
    assert {"success", "red_dropped", "blue_toppled", "unstable_robot", "time_out"} <= set(vars(terms))
    assert terms.time_out.time_out and not terms.success.time_out
    rewards = cfg.rewards
    weights = {name: term.weight for name, term in vars(rewards).items() if term is not None}
    assert weights == {
        "approach_progress": 1.0,
        "held_goal": 0.1,
        "milestones": 10.0,
        "success": 200.0,
        "cube_lost": -50.0,
        "action_rate": -0.002,
        "joint_velocity": -0.0002,
        "joint_limits": -1.0,
    }
    # Every term that loses a cube is penalized, and none of them is a time-out.
    assert set(rewards.cube_lost.params["term_keys"]) == {
        "red_dropped",
        "blue_toppled",
        "cube_a_out_of_bounds",
        "cube_b_out_of_bounds",
    }


def test_cube_teacher_events_and_sim2real():
    from isaaclab_tutorial.assets import CUBE_RESET_DATASET
    from isaaclab_tutorial.tasks.place_cube.config.so101.env_cfg import Sim2RealEventsCfg
    from isaaclab_tutorial.tasks.place_cube.reset.curriculum import ALL_PHASES as CUBE_ALL
    from isaaclab_tutorial.tasks.place_cube.reset.curriculum import CANONICAL_START as CUBE_START

    nominal, sim2real = _cube(), _cube("sim2real")
    reset = nominal.events.reset_from_dataset.params
    assert reset["dataset_path"] == str(CUBE_RESET_DATASET) and reset["phase_weights"] == CUBE_ALL
    assert nominal.events.cube_a_mass.mode == "startup"
    assert nominal.events.cube_a_mass.params["mass_distribution_params"] == (0.015, 0.025)
    assert nominal.events.cube_a_material.params["static_friction_range"] == (0.7, 1.3)
    assert isinstance(sim2real.events, Sim2RealEventsCfg)
    assert sim2real.events.reset_from_dataset.params["phase_weights"] == CUBE_START
    assert sim2real.events.reset_from_dataset.params["home_min_cube_gap"] >= 0.0425
    assert sim2real.events.cube_a_mass.mode == "reset"
    assert sim2real.rewards.jaw_support_force.weight == -8.0
    assert sim2real.rewards.approach_progress.params["height_offset"] == 0.008


@pytest.mark.parametrize("presets", [(), ("sim2real",)])
def test_cube_play_uses_home_starts_and_pins_cubes_only_without_sim2real(monkeypatch, presets):
    monkeypatch.setattr(evaluation, "EXACT_EVALUATION_ACTIVE", False)
    cfg = _cube(*presets)
    cfg.play_mode()
    from isaaclab_tutorial.tasks.place_cube.reset.curriculum import CANONICAL_START as CUBE_START

    assert cfg.events.reset_from_dataset.params["sequential"]
    assert cfg.events.reset_from_dataset.params["phase_weights"] == CUBE_START
    pinned = cfg.events.cube_a_material.params["static_friction_range"] == (0.7, 0.7)
    assert pinned == (not presets)


def test_cube_teacher_ppo_mirrors_the_vial_teacher():
    from isaaclab_tutorial.tasks.place_cube.config.so101.agents.rsl_rl_ppo_cfg import SO101CubeStatePPORunnerCfg

    cube, vial = SO101CubeStatePPORunnerCfg(), SO101StatePPORunnerCfg()
    assert cube.experiment_name == "so101_cube_state" and cube.max_iterations == 1500
    assert cube.obs_groups == vial.obs_groups == {"actor": ["policy"], "critic": ["critic"]}
    assert cube.actor.to_dict() == vial.actor.to_dict() and cube.critic.to_dict() == vial.critic.to_dict()
    assert cube.algorithm.to_dict() == vial.algorithm.to_dict()
