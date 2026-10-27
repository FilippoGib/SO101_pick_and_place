"""Record what the wrist-camera student sees during ``play``, as one tiled video next to the scene clip.

Passing ``--external_callback isaaclab_tutorial.utils.camera_video.install_camera_video`` (with ``--video``) writes
``wrist_<checkpoint>_<index>.mp4`` beside the scene clips. Tile *k* is environment *k*'s newest ``wrist_rgb`` frame,
exactly the student's input. Its border turns green when the episode ends in success, yellow on a time-out and red on
any other termination, for one second. The clips use the scene recorder's length, interval and fps, so frame *i* of
both videos comes from the same environment step. ``SO101_CAMERA_VIDEO_ENVS`` sets how many environments are shown
(default 16).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

TILE_SCALE = 3
"""Each 64 x 48 student image is shown at 192 x 144."""

BORDER = 4
"""Border and gap width [px]."""

OUTCOME_HOLD_STEPS = 30
"""Steps an episode's outcome color stays on its tile (one second at 30 Hz)."""

OUTCOME_COLORS = np.array([[0, 0, 0], [40, 200, 70], [240, 200, 30], [220, 40, 40]], dtype=np.uint8)
"""Border colors per outcome code: running, success, time-out, other termination."""

DEFAULT_MAX_ENVS = 16


def tile_frames(images: np.ndarray, outcomes: np.ndarray, columns: int | None = None) -> np.ndarray:
    """Lay out ``(N, H, W, 3)`` uint8 images in a grid with outcome-colored borders and env-index labels."""
    from PIL import Image, ImageDraw

    count, height, width, _ = images.shape
    columns = columns or math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    images = images.repeat(TILE_SCALE, axis=1).repeat(TILE_SCALE, axis=2)
    tile_h, tile_w = height * TILE_SCALE, width * TILE_SCALE
    # Each cell is the image, a border ring around it, and a gap to the next cell.
    stride_h, stride_w = tile_h + 3 * BORDER, tile_w + 3 * BORDER
    grid = np.full((rows * stride_h + BORDER, columns * stride_w + BORDER, 3), 24, dtype=np.uint8)
    corners = [((i // columns) * stride_h + BORDER, (i % columns) * stride_w + BORDER) for i in range(count)]
    for index, (top, left) in enumerate(corners):
        grid[top : top + tile_h + 2 * BORDER, left : left + tile_w + 2 * BORDER] = OUTCOME_COLORS[outcomes[index]]
        grid[top + BORDER : top + BORDER + tile_h, left + BORDER : left + BORDER + tile_w] = images[index]
    canvas = Image.fromarray(grid)
    draw = ImageDraw.Draw(canvas)
    for index, (top, left) in enumerate(corners):
        top, left = top + BORDER, left + BORDER
        draw.rectangle((left, top, left + 22, top + 12), fill=(0, 0, 0))
        draw.text((left + 2, top + 1), str(index), fill=(255, 255, 255))
    # libx264 with yuv420p needs even frame dimensions.
    frame = np.asarray(canvas)
    return frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]


def outcome_codes(terminated: torch.Tensor, time_outs: torch.Tensor, success: torch.Tensor | None) -> torch.Tensor:
    """Outcome code per environment for this step: 0 running, 1 success, 2 time-out, 3 other termination."""
    codes = torch.zeros_like(terminated, dtype=torch.long)
    codes[terminated] = 3
    codes[time_outs] = 2
    if success is not None:
        codes[success & terminated] = 1
    return codes


def _grid_recorder_class():
    from isaaclab.envs.utils.video_recorder import VideoRecorder

    class WristGridRecorder(VideoRecorder):
        """Isaac Lab's clip recorder, fed the tiled student input instead of a visualizer or sensor."""

        frame: np.ndarray | None = None

        def _get_frame(self) -> np.ndarray | None:
            return self.frame

    return WristGridRecorder


def _recorder_cfg(env):
    """Copy the scene recorder's timing and output folder, writing ``wrist_*`` clips."""
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    scene = list(getattr(env.cfg, "video_recorders", None) or [])
    if scene:
        cfg = scene[0].replace(source="sensor:wrist_camera")
        prefix = cfg.output_filename_prefix
        cfg.output_filename_prefix = (
            "wrist" + prefix.removeprefix("clip") if prefix.startswith("clip") else f"wrist_{prefix}"
        )
    else:
        log_dir = getattr(env.cfg, "log_dir", None) or "."
        cfg = VideoRecorderCfg(
            source="sensor:wrist_camera",
            output_dir=os.path.join(log_dir, "videos", "play"),
            output_filename_prefix="wrist",
        )
        print("[camera_video] No --video recorder configured; recording 200-step wrist clips only.", flush=True)
    return cfg


def install_camera_video() -> list[str]:
    """Record the tiled student input of the first ``SO101_CAMERA_VIDEO_ENVS`` environments during play."""
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    max_envs = int(os.environ.get("SO101_CAMERA_VIDEO_ENVS", DEFAULT_MAX_ENVS))
    original_step, original_close = RslRlVecEnvWrapper.step, RslRlVecEnvWrapper.close
    state: dict = {}

    def recorded_step(self, actions):
        obs, rewards, dones, extras = original_step(self, actions)
        env = self.unwrapped
        if "recorder" not in state:
            state["recorder"] = _grid_recorder_class()(_recorder_cfg(env), env)
            state["hold"] = torch.zeros(self.num_envs, dtype=torch.long, device=env.device)
            state["outcome"] = torch.zeros_like(state["hold"])
        terms = env.termination_manager
        success = terms.get_term("success") if "success" in terms.active_terms else None
        codes = outcome_codes(terms.terminated, terms.time_outs, success)
        ended = codes > 0
        state["hold"] = (state["hold"] - 1).clamp_min(0)
        state["outcome"][ended] = codes[ended]
        state["hold"][ended] = OUTCOME_HOLD_STEPS
        shown = min(max_envs, self.num_envs)
        # wrist_rgb stacks frames oldest first along channels; the newest is the last three.
        images = obs["wrist_rgb"][:shown, -3:].permute(0, 2, 3, 1).clamp(0, 1).mul(255).byte().cpu().numpy()
        visible = torch.where(state["hold"] > 0, state["outcome"], 0)[:shown].cpu().numpy()
        recorder = state["recorder"]
        recorder.frame = tile_frames(images, visible)
        recorder.step()
        return obs, rewards, dones, extras

    def recorded_close(self):
        if "recorder" in state:
            state["recorder"].close()
        return original_close(self)

    RslRlVecEnvWrapper.step = recorded_step
    RslRlVecEnvWrapper.close = recorded_close
    return sys.argv[1:]


def install_camera_video_and_episode_counter() -> list[str]:
    """:func:`install_camera_video` plus the exact episode counter of :mod:`isaaclab_tutorial.utils.evaluation`."""
    from isaaclab_tutorial.utils import evaluation

    install_camera_video()
    return evaluation.install_episode_counter()
