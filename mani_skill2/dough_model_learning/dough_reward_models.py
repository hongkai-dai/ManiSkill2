from typing import Tuple, Dict

import torch

from mani_skill2.dynamics.reward import GoalBasedRewardModel

# TODO(blake.wulfe): Where to put this?
class FlatDoughRollingRewardModel(GoalBasedRewardModel):
    def step(
        self, state: torch.Tensor, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[float, Dict]:
        dims = tuple(range(1, state.ndim))
        num_pos = (state > 0).sum(dim=dims)
        avg_pos = state.sum(dim=dims) / (num_pos + 1e-12)
        rew = -avg_pos * 10
        return rew, {}

    def set_goal(self, goal):
        pass
