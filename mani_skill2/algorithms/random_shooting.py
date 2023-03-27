from typing import Callable, Tuple

import numpy as np
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
        device: str = "cuda",
        verbose_info: bool = False,
    ):
        """
        Args:
            generative_env: The environment that wraps the dynamics and the reward.
            action_sampler: action_sampler(obs, steps) returns a batch of action
                sequence, each sequence contains `steps` actions. Namely
                the returned value has shape (num_samples, steps, act_size)
            planning_steps: The number of steps in the planning horizon.
            discount_factor: How much to exponentially discount future rewards.
            device: The device on which to place tensors created in this class.
                TODO(blake.wulfe): Once PlanningAgent is implemented, remove this.
            verbose_info: If True, stores a large amount of information in the `info`
                dict returned from step.
        """
        self.generative_env = generative_env
        self.action_sampler = action_sampler
        self.planning_steps = planning_steps
        self.discount_factor = discount_factor
        self.device = device
        self.verbose_info = verbose_info

    def step(
        self,
        obs: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """Plans the best action sequence.

        Given the current observation, sample many action sequence. Compute the
        cumulative rewards, choose the best action sequence with the highest reward.

        Args:
            obs: The current observation.

        Returns:
            action (torch.Tensor): The next action.
            info (dict): the auxillary information.
        """
        action_sequences = self.action_sampler(obs, self.planning_steps)
        action_sequences = action_sequences.to(self.device)

        num_rollouts = action_sequences.shape[0]
        total_rewards = torch.zeros((num_rollouts,), device=self.device)
        with torch.no_grad():
            initial_state = self.generative_env.dynamics_model.state_from_observation(
                obs
            )
            initial_state = torch.tensor(initial_state, device=self.device)
            state_rollouts = torch.empty(
                [num_rollouts, self.planning_steps + 1] + list(initial_state.shape),
                device=self.device,
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
                total_rewards += self.discount_factor ** i * rewards
            best_reward, best_rollout_index = total_rewards.max(dim=0)

            act_sequence = action_sequences[best_rollout_index]
            state_sequence = state_rollouts[best_rollout_index]

            info = dict(
                action_sequence=act_sequence,
                best_reward=best_reward,
            )
            if self.verbose_info:
                info["state_sequence"] = state_sequence

            action = act_sequence[0].detach().cpu().numpy()
            return action, info
