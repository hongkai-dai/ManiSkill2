import abc
from typing import Dict, Tuple

import gym
import torch

from mani_skill2.dynamics.base import (
    DynamicsModel,
)

from mani_skill2.dynamics.reward import (
    GoalBasedRewardModel,
)


class GenerativeEnv(abc.ABC):
    dynamics_model: DynamicsModel
    reward_model: GoalBasedRewardModel

    """
    Abstract base class that planners take as input. This class wraps the 
    dynamics model together with the reward.
    """

    @abc.abstractmethod
    def reset(self) -> torch.Tensor:
        """
        Reset and return the state.
        """

    def step(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, Dict]:
        """
        Steps the dynamical system for one transition.

        Args:
            state (torch.Tensor): the current state of the dynamical system.
            action (torch.Tensor): the control action of the dynamical stystem.

        Returns:
            next_state (torch.Tensor): The next state after the transition.
            reward (torch.Tensor): The reward obtained in this one transition.
            terminated (bool): terminate the roll out or not.
            info (Dict):  auxiliary diagnostic information.
        """
        next_state, terminated, info = self.dynamics_model.step(state, action)
        obs = self.observation(state)
        reward, _ = self.reward_model.step(state, obs, action)
        return next_state, reward, terminated, info

    @abc.abstractmethod
    def observation(self, state: torch.Tensor) -> torch.Tensor:
        """
        Generate observation for a batch of state.

        Args:
            state (torch.Tensor): a batch of state whose observation will be computed.

        Returns:
            obs (torch.Tensor): the observation of that batch of state.
        """

    @abc.abstractmethod
    def observation_space(self) -> gym.Space:
        """
        Returns the observation space of the environment.
        """

    @abc.abstractmethod
    def action_space(self) -> gym.Space:
        """
        Returns the action space of the environment.
        """
