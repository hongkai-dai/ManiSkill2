from collections.abc import Callable
from typing import Tuple

import torch

from mani_skill2.dynamics.generative_env import GenerativeEnv


class RandomShootingAgent:
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
        action_sequences = self.action_sampler(obs, self.planning_steps)
        num_rollouts = action_sequences.shape[0]
        total_rewards = torch.zeros((num_rollouts,))
        state_rollouts = []
        with torch.no_grad():
            initial_state = self.generative_env.dynamics_model.state_from_observation(
                obs
            )
            # Repeat initial_state along dim 0 for num_rollouts times.
            # Is this the best way to repeat the state?
            states = initial_state.unsqueeze(0).repeat(
                [num_rollouts] + [1] * len(initial_state.shape)
            )
            state_rollouts.append(states)
            # Compute pow(discount_factor, steps)
            gamma_power = 1.0
            for i in range(self.planning_steps):
                new_states, rewards, terminated, info = self.generative_env.step(
                    states, action_sequences[:, i, :]
                )
                total_rewards += gamma_power * rewards
                gamma_power *= self.discount_factor
                state_rollouts.append(new_states)
                states = new_states
            best_reward, best_rollout_index = total_rewards.max(dim=0)

            act_sequence = action_sequences[best_rollout_index]
            state_sequence = torch.cat(
                [
                    state_rollout[best_rollout_index].unsqueeze(0)
                    for state_rollout in state_rollouts
                ],
                dim=0,
            )

            return act_sequence, state_sequence, best_reward
