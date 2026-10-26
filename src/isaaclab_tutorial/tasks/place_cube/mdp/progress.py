"""Episode-local cube placement milestones."""

import torch


class PlacementProgress:
    def __init__(self, num_envs, device, stable_steps=10, grasp_steps=2):
        self.stable_steps = stable_steps
        self.grasp_steps = grasp_steps
        self.grasped = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.lifted = torch.zeros_like(self.grasped)
        self.inside = torch.zeros_like(self.grasped)
        self.success = torch.zeros_like(self.grasped)
        self.tipped_cup = torch.zeros_like(self.grasped)
        self.grasp_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.stable_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.time_to_success = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for value in (self.grasped, self.lifted, self.inside, self.success, self.tipped_cup):
            value[ids] = False
        self.grasp_count[ids] = 0
        self.stable_count[ids] = 0
        self.time_to_success[ids] = -1

    def update(self, holding, lifted, inside, seated, upright, step):
        self.grasp_count.copy_(torch.where(holding, self.grasp_count + 1, torch.zeros_like(self.grasp_count)))
        self.grasped |= self.grasp_count >= self.grasp_steps
        self.lifted |= lifted
        self.inside |= inside
        self.tipped_cup |= ~upright
        self.stable_count.copy_(torch.where(seated, self.stable_count + 1, torch.zeros_like(self.stable_count)))
        newly_successful = ~self.success & (self.stable_count >= self.stable_steps)
        self.time_to_success.copy_(torch.where(newly_successful, step, self.time_to_success))
        self.success |= newly_successful
        return self.success.clone()


def placement_progress(env):
    if not hasattr(env, "_so101_placement_progress"):
        env._so101_placement_progress = PlacementProgress(env.num_envs, env.device)
    return env._so101_placement_progress
