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

    def rollout(
        self,
        state_init: torch.Tensor,
        act_sequence: torch.Tensor,
        discount_factor: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given the initial state and the action sequence, compute the rollout reward and state/observation sequence.

        Args:
            state_init (torch.Tensor): a batch of initial state, size is batch_size * state_size.
            act_sequence (torch.Tensor): a batch of action sequences, size is batch_size * num_steps * act_size.
            discount_factor (float): discount the transition step by this factor per step.

        Returns:
            state_sequence (torch.Tensor): a batch of state sequences in the rollouts.
            rewards (torch.Tensor): the total cumulative reward for each rollout.
        """
        device = state_init.device
        num_rollouts = state_init.shape[0]
        assert num_rollouts == act_sequence.shape[0]
        num_steps = act_sequence.shape[1]
        state_sequence = torch.empty(
            [num_rollouts, num_steps + 1] + list(state_init.shape)[1:], device=device
        )
        state_sequence[:, 0, ...] = state_init
        rewards = torch.empty(num_rollouts, device=device)
        for i in range(num_steps):
            state_sequence[:, i + 1, :], reward, _, _ = self.step(
                state_sequence[:, i, :], act_sequence[:, i, :]
            )
            rewards += discount_factor**i * reward
        return state_sequence, rewards
