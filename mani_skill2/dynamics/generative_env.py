import abc
from typing import Dict, Tuple

import gym
import torch

from mani_skill2.dynamics.base import (
    DynamicsModel,
    ActType,
    ObsType,
    StateType,
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
    def reset(self) -> StateType:
        """
        Reset and return the state.
        """

    def step(
        self, state: StateType, action: ActType
    ) -> Tuple[StateType, torch.Tensor, bool, Dict]:
        """
        Steps the dynamical system for one transition.

        Args:
            state (StateType): the current state of the dynamical system.
            action (ActType): the control action of the dynamical stystem.

        Returns:
            next_state (StateType): The next state after the transition.
            reward (torch.Tensor): The reward obtained in this one transition.
            terminated (bool): terminate the roll out or not.
            info (Dict):  auxiliary diagnostic information.
        """
        next_state, terminated, info = self.dynamics_model.step(state, action)
        obs = self.observation(state)
        next_obs = self.observation(next_state)
        reward, _ = self.reward_model.step(state, obs, next_state, next_obs, action)
        return next_state, reward, terminated, info

    @abc.abstractmethod
    def observation(self, state: StateType) -> ObsType:
        """
        Generate observation for a state.

        Args:
            state (StateType): a state whose observation will be computed.

        Returns:
            obs (ObsType): the observation of that state.
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
