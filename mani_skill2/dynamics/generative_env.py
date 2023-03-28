import abc
from typing import Dict, Tuple

import gym
import torch

from enum import Enum

from mani_skill2.dynamics.base import (
    DynamicsModel,
)

from mani_skill2.dynamics.reward import (
    GoalBasedRewardModel,
)


class RewardOption(Enum):
    FinalTimeOnly = 1  # Only impose the (un-discounted) reward at the final time.
    Always = 2  # Impose the discounted reward in the entire rollout.


# TODO(blake.wulfe): Decide if this should be an ABC or not.
class GenerativeEnv:
    def __init__(
        self,
        dynamics_model,
        reward_model,
        observation_space,
        action_space,
        reward_option: RewardOption = RewardOption.Always,
    ):
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_option = reward_option

    @abc.abstractmethod
    def reset(self) -> torch.Tensor:
        """
        Reset and return the state.
        """
        pass

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
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Given the initial state and the action sequence, compute the rollout reward and state/observation sequence.

        Args:
            state_init (torch.Tensor): a batch of initial state, size is batch_size * state_size.
            act_sequence (torch.Tensor): a batch of action sequences, size is batch_size * num_steps * act_size.
            discount_factor (float): discount the transition step by this factor per step.

        Returns:
            state_sequence (torch.Tensor): a batch of state sequences in the rollouts,
                size is batch_size * (num_steps+1) * state_size.
            rewards (torch.Tensor): the total cumulative reward for each rollout,
                size is batch_size
            info (Dict): auxillary information for debugging.
        """
        device = state_init.device
        num_rollouts = state_init.shape[0]
        assert num_rollouts == act_sequence.shape[0]
        num_steps = act_sequence.shape[1]
        state_sequence = torch.empty(
            [num_rollouts, num_steps + 1] + list(state_init.shape)[1:], device=device
        )
        state_sequence[:, 0, ...] = state_init
        rewards = torch.zeros(num_rollouts, device=device)
        dyn_infos = []
        for i in range(num_steps):
            state_sequence[:, i + 1, :], reward, _, step_info = self.step(
                state_sequence[:, i, :], act_sequence[:, i, :]
            )
            if self.reward_option == RewardOption.Always:
                rewards += discount_factor**i * reward
            elif self.reward_option == RewardOption.FinalTimeOnly:
                rewards += reward if i == num_steps - 1 else 0
            dyn_infos.append(step_info)
        return state_sequence, rewards, dict(dyn_infos=dyn_infos)
