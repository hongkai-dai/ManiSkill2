from typing import Dict, Tuple

import gym
import numpy as np
from ray.rllib.policy.sample_batch import SampleBatch
import torch
import torch.nn as nn

from mani_skill2.dynamics.generative_env import GenerativeEnv, RewardOption
from mani_skill2.dynamics.network_dynamics_model import NetworkDynamicsModel
from mani_skill2.dynamics.normalizers import Normalizer
from mani_skill2.dynamics.reward import GoalBasedRewardModel


class MockFCNetwork(nn.Module):
    def __init__(self, state_size: int, act_size: int):
        super().__init__()
        self.layer = nn.Linear(state_size + act_size, state_size)

    def forward(self, state: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat((state, act), dim=-1)
        x = self.layer(x)
        return x


def get_mock_fc_batch(
    batch_size: int,
    state_size: int,
    act_size: int,
):
    state = torch.zeros((batch_size, state_size))
    act = torch.zeros((batch_size, act_size))
    rew = torch.zeros((batch_size,))
    new_state = torch.zeros((batch_size, state_size))
    dones = torch.zeros((batch_size,), dtype=bool)
    return {
        SampleBatch.OBS: state,
        SampleBatch.ACTIONS: act,
        SampleBatch.REWARDS: rew,
        SampleBatch.NEXT_OBS: new_state,
        SampleBatch.DONES: dones,
    }


class MockRewardModel(GoalBasedRewardModel):
    def __init__(self, state_size: int, act_size: int):
        super().__init__()
        self.state_size = state_size
        self.act_size = act_size
        self.register_buffer("goal", torch.ones(state_size))

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

    def __init__(
        self, state_size, act_size, device="cpu", reward_option=RewardOption.Always
    ):
        self.dynamics_model = NetworkDynamicsModel(
            MockFCNetwork(state_size, act_size), is_residual=True
        )
        self.dynamics_model.to(device)
        self.reward_model = MockRewardModel(state_size, act_size)
        self.reward_model.to(device)
        self.state_size = state_size
        self.act_size = act_size
        self.reward_option = reward_option

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


class MockNormalizer(Normalizer):
    def normalize(self, x):
        return x * 10

    def denormalize(self, x):
        return x / 10
