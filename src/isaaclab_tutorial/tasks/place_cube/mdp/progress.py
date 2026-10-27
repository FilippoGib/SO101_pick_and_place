"""Episode-local stacking milestones with safe partial-reset semantics."""

from collections.abc import Sequence

import torch


class StackProgress:
    """Per-episode milestones, each latched the first time it is observed.

    * ``grasped``: both jaws hold the red cube off the desk for ``grasp_steps`` consecutive steps,
    * ``lifted``: the red cube's lowest corner has cleared the blue cube's top face,
    * ``placed``: the red cube has rested on the blue cube's top face,
    * ``success``: the released red cube has sat centered and flat on the upright blue cube for ``stable_steps``
      consecutive steps.
    """

    def __init__(self, num_envs: int, device: str | torch.device, stable_steps: int = 10, grasp_steps: int = 2):
        self.stable_steps = stable_steps
        self.grasp_steps = grasp_steps
        self.grasped = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.lifted = torch.zeros_like(self.grasped)
        self.placed = torch.zeros_like(self.grasped)
        self.success = torch.zeros_like(self.grasped)
        self.unsafe_stack_impact = torch.zeros_like(self.grasped)
        self.grasp_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.stable_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.time_to_success = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    def reset(self, env_ids: Sequence[int] | slice | torch.Tensor | None = None) -> None:
        """Reset only the requested environments."""
        ids = slice(None) if env_ids is None else env_ids
        self.grasped[ids] = False
        self.lifted[ids] = False
        self.placed[ids] = False
        self.success[ids] = False
        self.unsafe_stack_impact[ids] = False
        self.grasp_count[ids] = 0
        self.stable_count[ids] = 0
        self.time_to_success[ids] = -1

    def update(
        self,
        holding: torch.Tensor,
        cleared: torch.Tensor,
        placed: torch.Tensor,
        stacked: torch.Tensor,
        step: torch.Tensor,
    ) -> torch.Tensor:
        """Latch milestones from the current physical state and return confirmed success."""
        self.grasp_count.copy_(torch.where(holding, self.grasp_count + 1, torch.zeros_like(self.grasp_count)))
        self.grasped |= self.grasp_count >= self.grasp_steps
        # A lift or placement only counts for a cube the robot has actually grasped.
        self.lifted |= cleared & self.grasped
        self.placed |= placed & self.grasped
        self.stable_count.copy_(torch.where(stacked, self.stable_count + 1, torch.zeros_like(self.stable_count)))
        newly_successful = ~self.success & self.grasped & (self.stable_count >= self.stable_steps)
        self.time_to_success.copy_(torch.where(newly_successful, step, self.time_to_success))
        self.success |= newly_successful
        return self.success.clone()


def stack_progress(env) -> StackProgress:
    """Return the episode progress shared by the success termination, rewards and observations."""
    if not hasattr(env, "_so101_stack_progress"):
        env._so101_stack_progress = StackProgress(env.num_envs, env.device)
    return env._so101_stack_progress
