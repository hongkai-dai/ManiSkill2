from collections.abc import Callable
from typing import Tuple

import torch

from mani_skill2.dynamics.generative_env import GenerativeEnv
from mani_skill2.algorithms.gym_agent import GymAgent


class RandomShootingAgent(GymAgent):
    def __init__(
        self,
        generative_env: GenerativeEnv,
        action_sampler: Callable[[torch.Tensor, int], torch.Tensor],
        planning_steps: int,
        discount_factor: float = 1.0,
    ):
        """
        Args:
            generative_env: The environment that wraps the dynamics and the reward.
            action_sampler: action_sampler(obs, steps) returns a batch of action
                            sequence, each sequence contains `steps` actions. Namely
                            the returned value has shape (num_samples, steps, act_size)
            planning_steps: The number of steps in the planning horizon.
        """
        self.generative_env = generative_env
        self.action_sampler = action_sampler
        self.planning_steps = planning_steps
        self.discount_factor = discount_factor

    def step(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Plans the best action sequence.

        Given the current observation, sample many action sequence. Compute the
        cumulative rewards, choose the best action sequence with the highest reward.

        Args:
            obs: The current observation

        Returns:
            act_sequence (torch.Tensor): The best action sequence.
            state_sequence (torch.Tensor): The state sequence in the best rollout.
            total_reward (torch.Tensor): The total reward of the best rollout.
        """
        device = obs.device
        action_sequences = self.action_sampler(obs, self.planning_steps)
        num_rollouts = action_sequences.shape[0]
        total_rewards = torch.zeros((num_rollouts,))
        with torch.no_grad():
            initial_state = self.generative_env.dynamics_model.state_from_observation(
                obs
            )
            state_rollouts = torch.empty(
                [num_rollouts, self.planning_steps + 1] + list(initial_state.shape),
                device=device,
            )
            # Prepend one dimension, repeat initial_state along this dimension for
            # num_rollouts times.
            state_rollouts[:, 0, ...] = initial_state.repeat(
                [num_rollouts] + [1] * initial_state.ndim
            )
            for i in range(self.planning_steps):
                (
                    state_rollouts[:, i + 1, ...],
                    rewards,
                    terminated,
                    info,
                ) = self.generative_env.step(
                    state_rollouts[:, i, ...], action_sequences[:, i, :]
                )
                total_rewards += self.discount_factor**i * rewards
            best_reward, best_rollout_index = total_rewards.max(dim=0)

            act_sequence = action_sequences[best_rollout_index]
            state_sequence = state_rollouts[best_rollout_index]

            return act_sequence, state_sequence, best_reward
