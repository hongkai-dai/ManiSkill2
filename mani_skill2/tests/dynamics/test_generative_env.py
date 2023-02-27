import pytest

import numpy as np
import torch

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
    def test_rollout(self, state_size, act_size, steps, batch_size, device):
        torch.random.manual_seed(123)
        generative_env = MockGenerativeEnv(state_size, act_size)
        generative_env.dynamics_model.to(device)
        generative_env.reward_model.to(device)

        discount_factor = 0.5
        state_init = torch.rand((batch_size, state_size), device=device)
        act_sequence = torch.rand((batch_size, steps, act_size), device=device)
        states, reward, info = generative_env.rollout(
            state_init, act_sequence, discount_factor
        )
        assert states.shape == (batch_size, steps + 1, state_size)
        np.testing.assert_allclose(
            states[:, 0, :].detach().cpu().numpy(), state_init.detach().cpu().numpy()
        )
        assert reward.shape == (batch_size,)
        assert isinstance(info, dict)
