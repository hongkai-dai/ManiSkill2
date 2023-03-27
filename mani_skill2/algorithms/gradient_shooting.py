import copy
from typing import Dict, Optional, Tuple

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
        action_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        device: str = "cuda",
        verbose_info: bool = False,
    ):
        """
        Args:
            generative_env: The environment that wraps the dynamics and reward.
            planning_steps: The number of steps in the planning horizon.
            discount_factor: discount the reward per step by this factor.
            optimizer: pytorch optimizer.
            gradient_steps: number of steps in gradient optimization.
            action_bounds: The lower and upper bound of the action.
            device: The device on which to place tensors created in this class.
                TODO(blake.wulfe): Once PlanningAgent is implemented, remove this.
            verbose_info: If True, stores a large amount of information in the `info`
                dict returned from step.
        """
        self.generative_env = generative_env
        self.planning_steps = planning_steps
        self.discount_factor = discount_factor
        self.optimizer = optimizer
        self.gradient_steps = gradient_steps
        self.action_bounds = action_bounds
        self.device = device
        self.verbose_info = verbose_info

    def set_action_sequences_init(self, act_sequences: torch.Tensor):
        """
        Set up the batch of initial guess of the action sequences. The optimization
        will start from this initial guess.

        Args:
            act_sequences: A batch of action sequences of size (batch_size, plannint_steps, act_size)
        """
        assert act_sequences.shape[1] == self.planning_steps
        self.act_sequences = act_sequences

    def _clip_actions(self, action_sequences: torch.Tensor) -> torch.Tensor:
        """
        Clip the action within the bounds.

        Args:
            action_sequences: Size (batch_size, planning_steps, act_size)
        Return:
            clipped_action_sequences: Size (batch_size, planning_steps, act_size)
        """
        if self.action_bounds is not None:
            return torch.clamp(
                action_sequences, self.action_bounds[0], self.action_bounds[1]
            )
        else:
            return action_sequences

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
            initial_states: Size of (batch_size, state_size). initial_states[i] is the
                initial state for the i'th rollout.
            act_sequences: Size of (batch_size,  planning_steps, act_size). act_sequences[i] is
                the action sequence for the i'th rollout.
            best_rewards: Size of batch_size. best_rewards[i] is the best reward for
                all previous attempted action sequences on initial_states[i]. After
                calling this function the best reward might be updated if this
                `act_sequences[i]` achieves better reward.
            best_act_sequences: Size of (batch_size, planning_steps, act_size). Stores
                the best action sequence for each initial state. If this
                `act_sequences[i]` improves the reward, then `best_act_sequences[i]`
                will be updated to `act_sequences[i]`.
        """
        batch_size = initial_states.shape[0]
        state_sequences = torch.empty(
            (batch_size, self.planning_steps + 1) + tuple(initial_states.shape[1:]),
            device=self.device,
        )
        total_rewards = torch.empty((batch_size,), device=self.device)
        dyn_info_list = []
        for i in range(batch_size):
            clipped_act_sequence = self._clip_actions(act_sequences[i])
            (
                state_sequences_i,
                total_rewards_i,
                dyn_info_i,
            ) = self.generative_env.rollout(
                initial_states[i].unsqueeze(0),
                clipped_act_sequence.unsqueeze(0),
                self.discount_factor,
            )
            state_sequences[i] = state_sequences_i.squeeze(0)
            total_rewards[i] = total_rewards_i.squeeze(0)
            dyn_info_list.append(dyn_info_i)
            if total_rewards[i] > best_rewards[i]:
                best_act_sequences[i] = act_sequences[i].clone()
                best_rewards[i] = total_rewards[i].clone()
            # We update the gradient for every action sequence, even if it does not
            # improve the reward, so as to avoid getting trapped in the local minimum.
            total_loss = -total_rewards_i
            total_loss.backward()

        return dict(dyn_infos=dyn_info_list, state_sequences=state_sequences)

    def step(self, obs: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Computes a batch of best action sequences for the observation using
        gradient-based shooting method.

        Args:
            obs: The current observations of size obs_size

        Returns:
            action: The immediate action of size act_size
            info: auxillary information for diagnostics.
        """

        batch_size = self.act_sequences.shape[0]
        self.act_sequences.requires_grad_(True)
        # Deep copy self.act_sequences so that this step() function does not modify the
        # class data.
        act_sequences = copy.deepcopy(self.act_sequences)
        optimizer = self.optimizer([act_sequences], lr=1e-1)
        initial_state = self.generative_env.dynamics_model.state_from_observation(
            obs
        ).to(self.device)
        initial_states = initial_state.repeat([batch_size] + [1] * initial_state.ndim)
        best_rewards = torch.full((batch_size,), -np.inf, device=self.device)
        best_act_sequences = torch.empty_like(act_sequences, device=self.device)
        rewards_curve = torch.empty(
            (self.gradient_steps, batch_size), device=self.device
        )
        act_sequences_all = torch.empty(
            (self.gradient_steps,) + tuple(act_sequences.shape), device=self.device
        )
        for gradient_step in range(self.gradient_steps):
            optimizer.zero_grad()
            act_sequences_all[gradient_step] = act_sequences.clone()
            dyn_infos = self._compute_rollout_gradient(
                initial_states, act_sequences, best_rewards, best_act_sequences
            )
            rewards_curve[gradient_step] = best_rewards.squeeze().detach()
            optimizer.step()
        best_act_sequences = self._clip_actions(best_act_sequences)
        info = dict(
            best_act_sequences=best_act_sequences,
            best_rewards=best_rewards,
            rewards_curve=rewards_curve,
            act_sequences_all=act_sequences_all,
        )
        if self.verbose_info:
            info["dyn_infos"] = dyn_infos
        # Choose the next action with the best reward
        action = (
            best_act_sequences[torch.argmax(best_rewards), 0, ...]
            .detach()
            .cpu()
            .numpy()
        )
        return action, info
