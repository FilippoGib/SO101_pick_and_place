"""Tiled student-camera video: grid layout, outcome borders, and the play hook."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab_rl import rsl_rl

from isaaclab_tutorial.utils import camera_video
from isaaclab_tutorial.utils.camera_video import BORDER, OUTCOME_COLORS, TILE_SCALE, outcome_codes, tile_frames


def _tile_origin(index, columns, height=48, width=64):
    stride_h, stride_w = height * TILE_SCALE + 3 * BORDER, width * TILE_SCALE + 3 * BORDER
    return (index // columns) * stride_h + BORDER, (index % columns) * stride_w + BORDER


def test_tiles_keep_env_order_scale_and_outcome_borders():
    images = np.zeros((5, 48, 64, 3), dtype=np.uint8)
    for index in range(5):
        images[index] = 40 * (index + 1)
    frame = tile_frames(images, np.array([0, 1, 2, 3, 0]))

    assert frame.shape[0] % 2 == 0 and frame.shape[1] % 2 == 0
    # Five tiles need a 3 x 2 grid.
    assert frame.shape[1] >= 3 * (64 * TILE_SCALE + 3 * BORDER)
    for index, outcome in enumerate([0, 1, 2, 3, 0]):
        top, left = _tile_origin(index, columns=3)
        # Bottom-right image pixel, away from the env-index label.
        bottom, right = top + BORDER + 48 * TILE_SCALE - 1, left + BORDER + 64 * TILE_SCALE - 1
        assert (frame[bottom, right] == 40 * (index + 1)).all()
        assert (frame[bottom + 1, right] == OUTCOME_COLORS[outcome]).all()


def test_outcome_codes_separate_success_time_out_and_failure():
    terminated = torch.tensor([False, True, True, False])
    time_outs = torch.tensor([False, False, False, True])
    success = torch.tensor([False, True, False, False])

    assert outcome_codes(terminated, time_outs, success).tolist() == [0, 1, 3, 2]
    assert outcome_codes(terminated, time_outs, None).tolist() == [0, 3, 3, 2]


class _Terminations:
    active_terms = ["success", "time_out"]

    def __init__(self):
        self.terminated = torch.tensor([False, False])
        self.time_outs = torch.tensor([False, False])
        self.success = torch.tensor([False, False])

    def get_term(self, name):
        assert name == "success"
        return self.success


class _FakeWrapper:
    def __init__(self):
        self.num_envs = 2
        self.unwrapped = self
        self.device = "cpu"
        self.cfg = SimpleNamespace(video_recorders=[], log_dir="unused")
        self.termination_manager = _Terminations()


class _FakeRecorder:
    def __init__(self, cfg, env):
        self.frames, self.closed = [], False

    def step(self):
        self.frames.append(self.frame)

    def close(self):
        self.closed = True


@pytest.fixture
def hooked(monkeypatch):
    images = torch.zeros(2, 6, 48, 64)
    images[:, 3:] = 1.0  # newest frame white, oldest black

    def step(self, actions):
        return {"wrist_rgb": images}, torch.zeros(2), torch.zeros(2), {}

    monkeypatch.setattr(rsl_rl.RslRlVecEnvWrapper, "step", step)
    monkeypatch.setattr(rsl_rl.RslRlVecEnvWrapper, "close", lambda self: None)
    monkeypatch.setattr(camera_video, "_grid_recorder_class", lambda: _FakeRecorder)
    monkeypatch.setattr(camera_video, "_recorder_cfg", lambda env: None)
    camera_video.install_camera_video()
    return rsl_rl.RslRlVecEnvWrapper


def test_play_hook_records_the_newest_frame_and_holds_the_outcome(hooked):
    env = _FakeWrapper()
    hooked.step(env, None)
    env.termination_manager.terminated = torch.tensor([True, False])
    env.termination_manager.success = torch.tensor([True, False])
    hooked.step(env, None)
    env.termination_manager.terminated = torch.tensor([False, False])
    for _ in range(camera_video.OUTCOME_HOLD_STEPS):
        hooked.step(env, None)
    hooked.close(env)

    frames = _recorded_frames(hooked)
    top, left = _tile_origin(0, columns=2)
    bottom, right = top + BORDER + 48 * TILE_SCALE - 1, left + BORDER + 64 * TILE_SCALE - 1
    assert (frames[0][bottom, right] == 255).all()
    border = [tuple(frame[bottom + 1, right]) for frame in frames]
    success = tuple(OUTCOME_COLORS[1])
    # Running, then success for OUTCOME_HOLD_STEPS frames, then running again.
    assert border[0] == tuple(OUTCOME_COLORS[0])
    assert border[1 : 1 + camera_video.OUTCOME_HOLD_STEPS] == [success] * camera_video.OUTCOME_HOLD_STEPS
    assert border[-1] == tuple(OUTCOME_COLORS[0])


def _recorded_frames(wrapper_class):
    for cell in wrapper_class.step.__closure__:
        contents = cell.cell_contents
        if isinstance(contents, dict) and "recorder" in contents:
            assert contents["recorder"].closed
            return contents["recorder"].frames
    raise AssertionError("hook state not found")
