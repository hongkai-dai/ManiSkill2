from typing import Dict, Tuple

import numpy as np
import torch

from mani_skill2.dynamics.generative_env import GenerativeEnv
from mani_skill2.algorithms.gym_agent import GymAgent


class GradientShootingAgent(GymAgent):
    """Find the action sequence by minimizing the cumulative loss through gradient descent.

    In the optimization problem, the decision variable is the action sequence. We start
    with an initial guess of the action sequence and then reduce the cumulative loss by
    following the gradient direction.
    """

    act_sequences: torch.Tensor

    def __init__(
        self,
        generative_env: GenerativeEnv,
        planning_steps: int,
        discount_factor: float = 1.0,
        optimizer=torch.optim.Adam,
        gradient_steps: int = 10,
    ):
        """
        Args:
            generative_env: The environment that wraps the dynamics and reward.
            planning_steps: The number of steps in the planning horizon.
            discount_factor: discount the reward per step by this factor.
            optimizer: pytorch optimizer.
            gradient_steps: number of steps in gradient optimization.
        """
        self.generative_env = generative_env
        self.planning_steps = planning_steps
        self.discount_factor = discount_factor
        self.optimizer = optimizer
        self.gradient_steps = gradient_steps

    def set_action_sequences_init(self, act_sequences: torch.Tensor):
        """
        Set up the batch of initial guess of the action sequences. The optimization
        will start from this initial guess.

        Args:
            act_sequences: A batch of action sequences of size (batch_size, plannint_steps, act_size)
        """
        assert act_sequences.shape[1] == self.planning_steps
        self.act_sequences = act_sequences

    def _compute_rollout_gradient(
        self,
        initial_states: torch.Tensor,
        act_sequences: torch.Tensor,
        best_rewards: torch.Tensor,
        best_act_sequences: torch.Tensor,
    ) -> Dict:
        """
        Compute the gradient of cummulative cost for each rollout w.r.t the action
        sequence in each roll out.

        Args:
            best_rewards: Size of batch_size. best_rewards[i] is the best reward for
                all previous attempted action sequences on initial_states[i]. After
                calling this function the best reward might be updated if this
                `act_sequences[i]` achieves better reward.
            best_act_sequences: Size of batch_size x planning_steps x act_size. Stores
                the best action sequence for each initial state. If this
                `act_sequences[i]` improves the reward, then `best_act_sequences[i]`
                will be updated to `act_sequences[i]`.
        """
        batch_size = initial_states.shape[0]
        state_sequences, total_rewards, dyn_infos = self.generative_env.rollout(
            initial_states, act_sequences, self.discount_factor
        )
        improve_mask = total_rewards > best_rewards
        best_act_sequences[improve_mask] = act_sequences[improve_mask].clone()
        best_rewards[improve_mask] = total_rewards[improve_mask]
        total_loss = -total_rewards
        # TODO(hongkai.dai): using the more efficient approach described in https://pytorch.org/functorch/stable/notebooks/per_sample_grads.html
        for i in range(batch_size):
            # We update the gradient for every action sequence, even if it does not
            # improve the reward, so as to avoid getting trapped in the local minimum.
            total_loss[i].backward(retain_graph=True)

        return dict(dyn_infos=dyn_infos, state_sequences=state_sequences)

    def step(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Computes a batch of best action sequences for the observation using
        gradient-based shooting method.

        Args:
            obs: The current observations of size obs_size

        Returns:
            actions: The immediate actions of size batch_size x act_size
            info: auxillary information for diagnostics.
        """
        device = obs.device
        batch_size = self.act_sequences.shape[0]
        self.act_sequences.requires_grad_(True)
        # I can't use act_sequences in self.optimizer([act_sequences]) since
        # act_sequences is a non-leaf Tensor and the optimizer cannot optimize a
        # non-leaf tensor. Hence this function will modify self.act_sequences.
        # act_sequences = self.act_sequences.clone()
        optimizer = self.optimizer([self.act_sequences])
        initial_states = self.generative_env.dynamics_model.state_from_observation(
            obs.repeat([batch_size] + [1] * obs.ndim)
        )
        best_rewards = torch.full((batch_size,), -np.inf, device=device)
        best_act_sequences = torch.empty_like(self.act_sequences, device=device)
        for _ in range(self.gradient_steps):
            optimizer.zero_grad()
            dyn_infos = self._compute_rollout_gradient(
                initial_states, self.act_sequences, best_rewards, best_act_sequences
            )
            optimizer.step()
        info = dict(dyn_infos=dyn_infos, best_act_sequences=best_act_sequences)
        return best_act_sequences[:, 0, ...], info
