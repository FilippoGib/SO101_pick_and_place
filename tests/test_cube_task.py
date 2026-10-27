"""Cube-stacking task: geometry, milestones, rewards, terminations and the reset artifact."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import TerminationTermCfg

# Import the submodules by path: the lazy ``mdp`` package also re-exports Isaac Lab's same-named modules.
import isaaclab_tutorial.tasks.place_cube.mdp.observations as observations
import isaaclab_tutorial.tasks.place_cube.mdp.rewards as rewards
import isaaclab_tutorial.tasks.place_cube.mdp.terminations as terminations
from isaaclab_tutorial.tasks.place_cube.mdp.geometry import (
    CUBE_REST_HEIGHT,
    CUBE_SIZE,
    STACK_HEIGHT,
    cube_flatness,
    cube_lowest_offset,
    cubes_overlap_xy,
    resting_on,
    stack_offset,
    yaw_quat,
)
from isaaclab_tutorial.tasks.place_cube.mdp.progress import StackProgress
from isaaclab_tutorial.tasks.place_cube.reset.curriculum import ALL_PHASES, CANONICAL_START
from isaaclab_tutorial.tasks.place_cube.reset.dataset import PHASE_NAMES, load_reset_dataset, save_reset_dataset

BLUE = (0.21, 0.11, CUBE_REST_HEIGHT)
STACKED = (BLUE[0], BLUE[1], CUBE_REST_HEIGHT + STACK_HEIGHT)


def _tilt(angle: float) -> torch.Tensor:
    """XYZW quaternion tilted about world x."""
    return torch.tensor([math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)])


# --- geometry ---------------------------------------------------------------------------------------------------


def test_flatness_accepts_any_face_up_and_any_yaw():
    yaws = yaw_quat(torch.tensor([0.0, 0.4, math.pi / 4, 2.0]))
    torch.testing.assert_close(cube_flatness(yaws), torch.ones(4))
    on_side = _tilt(math.pi / 2).unsqueeze(0)
    assert cube_flatness(on_side).item() == pytest.approx(1.0, abs=1e-6)
    assert cube_flatness(_tilt(math.radians(30)).unsqueeze(0)).item() == pytest.approx(math.cos(math.radians(30)))


def test_lowest_corner_is_half_a_side_below_a_flat_cube_and_lower_when_tilted():
    assert cube_lowest_offset(yaw_quat(torch.tensor([0.7]))).item() == pytest.approx(-CUBE_SIZE / 2)
    on_edge = cube_lowest_offset(_tilt(math.pi / 4).unsqueeze(0)).item()
    assert on_edge == pytest.approx(-CUBE_SIZE / math.sqrt(2))


def test_stack_offset_is_zero_for_a_centered_flat_stack():
    offset = stack_offset(torch.tensor([STACKED]), torch.tensor([BLUE]))
    torch.testing.assert_close(offset, torch.zeros(1, 3))
    assert resting_on(torch.tensor([STACKED]), torch.tensor([BLUE]), 0.010, 0.004).item()
    off_center = torch.tensor([[STACKED[0] + 0.011, STACKED[1], STACKED[2]]])
    assert not resting_on(off_center, torch.tensor([BLUE]), 0.010, 0.004).item()


def test_reset_overlap_uses_the_center_gap():
    a = torch.tensor([[0.0, 0.0, 0.05], [0.0, 0.0, 0.05]])
    b = torch.tensor([[0.05, 0.0, 0.05], [0.07, 0.0, 0.05]])
    assert cubes_overlap_xy(a, b, 0.06).tolist() == [True, False]


# --- progress -----------------------------------------------------------------------------------------------------


def test_progress_latches_and_requires_a_grasp_before_lift_place_or_success():
    progress = StackProgress(2, "cpu", stable_steps=2, grasp_steps=2)
    step = torch.zeros(2, dtype=torch.long)
    yes, no = torch.tensor([True, True]), torch.tensor([False, False])
    # Cleared, placed and stacked without ever being grasped (e.g. pushed): nothing counts.
    for _ in range(3):
        assert not progress.update(no, yes, yes, yes, step).any()
    assert not (progress.lifted | progress.placed).any()
    holding = torch.tensor([True, False])
    progress.update(holding, no, no, no, step)
    progress.update(holding, yes, yes, no, step)
    assert progress.grasped.tolist() == [True, False]
    assert progress.lifted.tolist() == [True, False] and progress.placed.tolist() == [True, False]
    progress.update(no, no, no, yes, step + 7)
    assert progress.update(no, no, no, yes, step + 8).tolist() == [True, False]
    assert progress.time_to_success.tolist() == [8, -1]
    progress.reset(torch.tensor([0]))
    assert not progress.grasped.any() and progress.time_to_success.tolist() == [-1, -1]


# --- fake environment ---------------------------------------------------------------------------------------------


def _proxy(tensor):
    return SimpleNamespace(torch=tensor)


def _cube(positions, quats=None, lin_vel=None):
    positions = torch.tensor(positions, dtype=torch.float32)
    count = len(positions)
    quats = yaw_quat(torch.zeros(count)) if quats is None else quats
    return SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=_proxy(positions),
            root_quat_w=_proxy(quats),
            root_lin_vel_w=_proxy(torch.zeros(count, 3) if lin_vel is None else lin_vel),
            root_ang_vel_w=_proxy(torch.zeros(count, 3)),
        )
    )


def _environment(monkeypatch, red, blue, *, touching=False, openness=1.0, red_quat=None, blue_quat=None):
    count = len(red)

    class Scene(dict):
        sensors = {}
        env_origins = torch.zeros(count, 3)

    env = SimpleNamespace(
        scene=Scene(cube_a=_cube(red, red_quat), cube_b=_cube(blue, blue_quat)),
        num_envs=count,
        device="cpu",
        extras={},
        episode_length_buf=torch.zeros(count, dtype=torch.long),
        step_dt=1 / 30,
    )
    contact = torch.as_tensor(touching).expand(count).clone()
    monkeypatch.setattr(terminations, "jaw_contacts", lambda env: (contact, contact))
    monkeypatch.setattr(terminations, "contact_force", lambda env, name: torch.zeros(count))
    monkeypatch.setattr(terminations, "gripper_openness", lambda env: torch.full((count,), openness))
    return env


def _term(env, grasped=True):
    term = terminations.stack_success(TerminationTermCfg(func=terminations.stack_success), env)
    term.progress.grasped[:] = grasped
    return term


def test_success_needs_ten_released_stable_centered_steps(monkeypatch):
    red = [STACKED, (STACKED[0] + 0.009, STACKED[1], STACKED[2]), (STACKED[0] + 0.011, STACKED[1], STACKED[2])]
    env = _environment(monkeypatch, red, [BLUE] * 3)
    term = _term(env)
    for _ in range(9):
        assert not term(env).any()
    assert term(env).tolist() == [True, True, False], "within 10 mm of centered succeeds; 11 mm does not"
    assert term.progress.placed.tolist() == [True, True, True], "15 mm placement tolerance"
    assert env.extras["log"]["Metrics/success_rate"].item() == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "case",
    ["holding", "closed_gripper", "tilted_red", "toppled_blue", "lifted_blue", "moving", "never_grasped"],
)
def test_success_rejects_each_violated_condition(monkeypatch, case):
    kwargs = {}
    red, blue = [STACKED], [BLUE]
    if case == "holding":
        kwargs["touching"] = True
    elif case == "closed_gripper":
        kwargs["openness"] = 0.1
    elif case == "tilted_red":
        kwargs["red_quat"] = _tilt(math.radians(20)).unsqueeze(0)
    elif case == "toppled_blue":
        kwargs["blue_quat"] = _tilt(math.radians(30)).unsqueeze(0)
    elif case == "lifted_blue":
        blue = [(BLUE[0], BLUE[1], BLUE[2] + 0.02)]
        red = [(STACKED[0], STACKED[1], STACKED[2] + 0.02)]
    env = _environment(monkeypatch, red, blue, **kwargs)
    if case == "moving":
        env.scene["cube_a"].data.root_lin_vel_w = _proxy(torch.tensor([[0.06, 0.0, 0.0]]))
    term = _term(env, grasped=case != "never_grasped")
    for _ in range(12):
        assert not term(env).any(), case


def test_reset_restores_dataset_milestones(monkeypatch):
    env = _environment(monkeypatch, [STACKED, STACKED], [BLUE, BLUE])
    term = _term(env, grasped=False)
    env._so101_reset_grasped = torch.tensor([True, False])
    env._so101_reset_lifted = torch.tensor([True, False])
    term.reset(None)
    assert term.progress.grasped.tolist() == [True, False]
    assert term.progress.grasp_count.tolist() == [term.progress.grasp_steps, 0]
    assert term.progress.lifted.tolist() == [True, False]


def test_red_dropped_only_after_a_lift_and_not_when_stacked(monkeypatch):
    on_desk = (0.23, -0.02, CUBE_REST_HEIGHT)
    on_edge = (0.23, -0.02, 0.035 + CUBE_SIZE / math.sqrt(2))
    env = _environment(monkeypatch, [on_desk, on_edge, STACKED, on_desk], [BLUE] * 4)
    term = _term(env)
    term.progress.lifted[:] = torch.tensor([True, True, True, False])
    assert terminations.red_dropped(env).tolist() == [True, True, False, False]


def test_blue_toppled_allows_nudges_but_not_tipping_or_lifting(monkeypatch):
    blue = [BLUE, (BLUE[0] + 0.03, BLUE[1], BLUE[2]), BLUE, (BLUE[0], BLUE[1], BLUE[2] + 0.02)]
    quats = yaw_quat(torch.tensor([0.0, 0.5, 0.0, 0.0]))
    quats[2] = _tilt(math.radians(30))
    env = _environment(monkeypatch, [STACKED] * 4, blue, blue_quat=quats)
    assert terminations.blue_toppled(env).tolist() == [False, False, True, True]


def test_stack_features_and_progress_flags(monkeypatch):
    env = _environment(monkeypatch, [(STACKED[0] + 0.003, STACKED[1], STACKED[2] + 0.02)], [BLUE])
    term = _term(env)
    features = observations.stack_features(env)
    assert features.shape == (1, 8)
    torch.testing.assert_close(features[0, :3], torch.tensor([0.003, 0.0, 0.02]))
    assert features[0, 3].item() == pytest.approx(0.003, abs=1e-6)
    assert features[0, 4].item() == pytest.approx(0.02, abs=1e-6), "clearance of the red bottom above the blue top"
    term(env)
    # 20 mm of clearance over the blue top exceeds the 10 mm lift threshold; the cube is not resting on it.
    assert observations.progress_flags(env).tolist() == [[1.0, 1.0, 0.0]]


# --- rewards ------------------------------------------------------------------------------------------------------


def test_milestones_pay_once_and_do_not_pay_seeded_reset_state(monkeypatch):
    reward = rewards.milestones.__new__(rewards.milestones)
    reward.paid = torch.zeros((2, 3), dtype=torch.bool)
    reward.started = torch.zeros(2, dtype=torch.bool)
    progress = SimpleNamespace(
        grasped=torch.tensor([False, True]), lifted=torch.tensor([False, True]), placed=torch.tensor([False, False])
    )
    monkeypatch.setattr(rewards, "stack_progress", lambda env: progress)
    assert reward(None).tolist() == [0.0, 0.0], "milestones present at reset are not paid"
    progress.grasped[0] = True
    assert reward(None).tolist() == [1.0, 0.0]
    progress.lifted[0] = True
    progress.placed[:] = True
    assert reward(None).tolist() == [6.0, 4.0]
    assert reward(None).tolist() == [0.0, 0.0]


def test_held_goal_pays_near_the_stacked_pose_only_while_held_and_unplaced(monkeypatch):
    env = _environment(
        monkeypatch,
        [STACKED, (STACKED[0], STACKED[1], STACKED[2] + 0.1), STACKED],
        [BLUE] * 3,
    )
    progress = SimpleNamespace(grasped=torch.tensor([True, True, False]), placed=torch.tensor([False, False, False]))
    monkeypatch.setattr(rewards, "stack_progress", lambda env: progress)
    values = rewards.held_goal(env, std=0.10)
    assert values[0].item() == pytest.approx(math.exp(-0.001 / 0.10))
    assert values[1].item() == pytest.approx(math.exp(-0.099 / 0.10))
    assert values[2].item() == 0.0
    progress.placed[0] = True
    assert rewards.held_goal(env)[0].item() == 0.0


def test_approach_progress_pays_distance_covered_and_stops_after_the_grasp(monkeypatch):
    reward = rewards.approach_progress.__new__(rewards.approach_progress)
    reward.previous_distance = torch.full((2,), float("nan"))
    grasped = torch.tensor([False, False])
    target = torch.tensor([[0.25, 0.0, 0.0], [0.25, 0.0, 0.0]])
    monkeypatch.setattr(rewards, "stack_progress", lambda env: SimpleNamespace(grasped=grasped))
    monkeypatch.setattr(rewards, "grasp_center_w", lambda env: torch.zeros(2, 3))
    monkeypatch.setattr(rewards, "cube_grasp_point_w", lambda env: target)
    assert reward(None, scale=0.01).tolist() == [0.0, 0.0]
    target[:, 0] -= 0.005
    assert reward(None, scale=0.01).tolist() == pytest.approx([0.5, 0.5])
    grasped[1] = True
    target[:, 0] -= 0.03
    assert reward(None, scale=0.01).tolist() == [1.0, 0.0]


# --- reset artifact -----------------------------------------------------------------------------------------------


def _states(rows: int = 14) -> dict[str, torch.Tensor]:
    pose = torch.zeros((rows, 7))
    pose[:, 6] = 1.0
    phase = torch.arange(rows, dtype=torch.long) % len(PHASE_NAMES)
    return {
        "joint_position": torch.zeros((rows, 6)),
        "joint_target": torch.zeros((rows, 6)),
        "cube_a_pose": pose.clone(),
        "cube_b_pose": pose.clone(),
        "phase": phase,
        "difficulty": phase.float() / 6.0,
        "grasped": phase >= 2,
        "lifted": phase >= 3,
    }


def test_reset_dataset_round_trip_and_rejects_the_vial_format(tmp_path):
    path = tmp_path / "cube_resets.pt"
    written = save_reset_dataset(path, _states(), generator={"seed": 1}, validation={"physics": "newton"})
    loaded = load_reset_dataset(path)
    assert loaded["content_sha256"] == written["content_sha256"] and loaded["row_count"] == 14
    assert loaded["phase_names"] == PHASE_NAMES
    assert PHASE_NAMES == ("home", "approach", "grasp", "lift", "transport", "place", "release")

    from isaaclab_tutorial.tasks.place_vial.reset.dataset import save_reset_dataset as save_vial

    vial = _states(8)
    vial["vial_pose"] = vial.pop("cube_a_pose")
    del vial["cube_b_pose"]
    vial["phase"] = torch.arange(8)
    save_vial(tmp_path / "vial.pt", vial, generator={}, validation={})
    with pytest.raises(ValueError, match="so101-cube-stack-reset-dataset"):
        load_reset_dataset(tmp_path / "vial.pt")


def test_reset_dataset_rejects_unnormalized_cube_quaternions(tmp_path):
    states = _states()
    states["cube_b_pose"][0, 3:] = torch.tensor([0.0, 0.0, 0.0, 2.0])
    with pytest.raises(ValueError, match="cube_b_pose quaternions"):
        save_reset_dataset(tmp_path / "bad.pt", states, generator={}, validation={})


def test_curriculum_covers_every_phase():
    assert len(ALL_PHASES) == len(CANONICAL_START) == len(PHASE_NAMES) == 7
    assert CANONICAL_START[0] == 1.0 and sum(CANONICAL_START) == 1.0
