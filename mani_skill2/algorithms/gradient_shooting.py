import numpy as np
import torch

from mani_skill2.dynamics.generative_env import GenerativeEnv
from mani_skill2.algorithms.gym_agent import GymAgent


class GradientShootingAgent(GymAgent):
    """
    Find the action sequence by minimizing the cumulative loss through gradient descent.

    In the optimization problem, the decision variable is the action sequence. We start
    with an initial guess of the action sequence and then reduce the cumulative loss by
    following the gradient direction.
    """

    act_sequence: torch.Tensor

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

    def set_action_sequence_init(self, act_sequence: torch.Tensor):
        """
        Set up the initial guess of the action sequence. The optimization will start
        from this initial guess.
        """
        assert act_sequence.shape[0] == self.planning_steps
        self.act_sequence = act_sequence

    def step(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO(hongkai.dai): take a step for a batch of observations.
        device = obs.device
        self.act_sequence.requires_grad_(True)
        optimizer = self.optimizer([self.act_sequence])
        state = self.generative_env.dynamics_model.state_from_observation(obs)
        best_reward = torch.tensor(-np.inf, device=device)
        best_act_sequence = torch.empty_like(self.act_sequence, device=device)
        for _ in range(self.gradient_steps):
            optimizer.zero_grad()
            states, total_reward = self.generative_env.rollout(
                state.unsqueeze(0), self.act_sequence.unsqueeze(0), self.discount_factor
            )
            if total_reward > best_reward:
                best_act_sequence = self.act_sequence.clone()
                best_reward = total_reward
            total_loss = -total_reward[0]
            total_loss.backward()
            optimizer.step()
        return best_act_sequence
