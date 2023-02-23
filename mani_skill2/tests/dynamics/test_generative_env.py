import pytest
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
