import abc
from typing import Tuple, TypeVar, Dict

import torch

from mani_skill2.dynamics.base import (
    ActType,
    ObsType,
    StateType,
)

GoalType = TypeVar("GoalType")


class GoalBasedRewardModel(abc.ABC, torch.nn.Module):
    """
    Given a goal, compute the reward for one transition.
    """

    @abc.abstractmethod
    def step(
        self,
        state: StateType,
        obs: ObsType,
        next_state: StateType,
        next_obs: ObsType,
        action: ActType,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute the reward for one transition.

        Args:
            state (StateType): The state at the start of the transition.
            obs (ObsType): The observation at the start of the transition.
            next_state (StateType): The state at the end of the transition.
            next_obs (ObsType): The observation at the end of the transition.
            action (ActType): The action during the transition.

        Returns:
            reward (torch.Tensor): The reward for this transition.
            info (Dict): auxiliary diagnostic information.
        """

    @abc.abstractmethod
    def set_goal(self, goal: GoalType):
        """
        Set the goal to compute the reward.
        """
