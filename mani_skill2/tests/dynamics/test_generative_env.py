import pytest

import numpy as np
import torch

from mani_skill2.dynamics.generative_env import RewardOption
from mani_skill2.utils.testing.dynamics_test_utils import MockGenerativeEnv


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

    @pytest.mark.parametrize("state_size", (3,))
    @pytest.mark.parametrize("act_size", (3,))
    @pytest.mark.parametrize("steps", (1, 5))
    @pytest.mark.parametrize("batch_size", (1, 4))
    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    @pytest.mark.parametrize(
        "reward_option", (RewardOption.Always, RewardOption.FinalTimeOnly)
    )
    def test_rollout(
        self, state_size, act_size, steps, batch_size, device, reward_option
    ):
        torch.random.manual_seed(123)
        dut = MockGenerativeEnv(state_size, act_size, device, reward_option)
        dut.dynamics_model.to(device)
        dut.reward_model.to(device)

        discount_factor = 0.5
        state_init = torch.rand((batch_size, state_size), device=device)
        act_sequence = torch.rand((batch_size, steps, act_size), device=device)
        states, reward, info = dut.rollout(state_init, act_sequence, discount_factor)
        assert states.shape == (batch_size, steps + 1, state_size)
        np.testing.assert_allclose(
            states[:, 0, :].detach().cpu().numpy(), state_init.detach().cpu().numpy()
        )
        assert reward.shape == (batch_size,)
        reward_per_step = torch.zeros((batch_size, steps), device=device)
        for i in range(steps):
            reward_per_step[:, i], _ = dut.reward_model.step(
                states[:, i, :], dut.observation(states[:, i, :]), act_sequence[:, i, :]
            )
        if reward_option == RewardOption.Always:
            reward_expected = torch.sum(
                reward_per_step
                * torch.pow(
                    discount_factor * torch.ones((batch_size, steps), device=device),
                    torch.arange(0, steps, device=device).repeat((batch_size, 1)),
                ),
                dim=-1,
            )
        elif reward_option == RewardOption.FinalTimeOnly:
            reward_expected = reward_per_step[:, -1]
        np.testing.assert_allclose(
            reward.cpu().detach().numpy(),
            reward_expected.cpu().detach().numpy(),
            atol=1e-5,
        )

        assert isinstance(info, dict)
