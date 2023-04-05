import abc
from typing import Tuple, TypeVar, Dict

import torch


class GoalBasedRewardModel(abc.ABC, torch.nn.Module):
    """
    Given a goal, compute the reward for one transition.
    """

    @abc.abstractmethod
    def step(
        self,
        state: torch.Tensor,
        obs: torch.Tensor,
        next_state: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute the reward for one transition.

        Args:
            state (torch.Tensor): The state at the start of the transition.
            obs (torch.Tensor): The observation at the start of the transition.
            next_state (torch.Tensor): The state at the end of the transition.
            next_obs (torch.Tensor): The observation at the end of the transition.
            action (torch.Tensor): The action during the transition.

        Returns:
            reward (torch.Tensor): The reward for this transition.
            info (Dict): auxiliary diagnostic information.
        """

    @abc.abstractmethod
    def set_goal(self, goal: torch.Tensor):
        """
        Set the goal to compute the reward.
        """
