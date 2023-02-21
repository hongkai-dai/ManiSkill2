from typing import Dict, Tuple

import gym
import numpy as np
import pytest
import torch


from mani_skill2.dynamics.generative_env import GenerativeEnv
from mani_skill2.dynamics.network_dynamics_model import NetworkDynamicsModel
from mani_skill2.dynamics.reward import GoalBasedRewardModel
from mani_skill2.utils.testing.dynamics_test_utils import MockFCNetwork


class MockRewardModel(GoalBasedRewardModel):
    def __init__(self, state_size: int, act_size: int):
        self.state_size = state_size
        self.act_size = act_size
        self.goal = torch.ones(state_size)

    def step(
        self,
        state: torch.Tensor,
        obs: torch.Tensor,
        next_state: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        reward = ((self.goal - state) ** 2).sum(dim=-1) + (action**2).sum(dim=-1)
        return reward, dict()

    def set_goal(self, goal: torch.Tensor):
        self.goal = goal


class MockGenerativeEnv(GenerativeEnv):
    dynamics_model: NetworkDynamicsModel
    reward_model: MockRewardModel

    def __init__(self, state_size, act_size):
        self.dynamics_model = NetworkDynamicsModel(
            MockFCNetwork(state_size, act_size), is_residual=True
        )
        self.reward_model = MockRewardModel(state_size, act_size)
        self.state_size = state_size
        self.act_size = act_size

    def reset(self) -> torch.Tensor:
        return torch.zeros(self.state_size)

    def observation(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def observation_space(self) -> gym.Space:
        return gym.space.Box(
            low=-100 * np.ones((self.state_size,)),
            high=100 * np.ones((self.state_size,)),
        )

    def action_space(self) -> gym.Space:
        return gym.space.Box(
            low=-100 * np.ones((self.act_size,)), high=100 * np.ones((self.act_size,))
        )


class TestGenerativeEnv:
    @pytest.mark.parametrize("state_size", (1, 3))
    @pytest.mark.parametrize("act_size", (1, 2))
    @pytest.mark.parametrize("batch_size", (1, 5))
    def test_step(self, state_size, act_size, batch_size):
        dut = MockGenerativeEnv(state_size, act_size)
        state = torch.tensor(range(batch_size * state_size), dtype=torch.float).reshape(
            (batch_size, state_size)
        )
        action = torch.tensor(range(batch_size * act_size), dtype=torch.float).reshape(
            (batch_size, act_size)
        )
        next_state, reward, terminated, info = dut.step(state, action)
        assert next_state.shape == (batch_size, state_size)
        assert reward.shape == (batch_size,)
