import numpy as np
import pytest
import torch

from mani_skill2.algorithms.random_shooting import RandomShootingAgent
from mani_skill2.utils.testing.dynamics_test_utils import MockGenerativeEnv


class TestRandomShootingAgent:
    @pytest.mark.parametrize("state_size", (3,))
    @pytest.mark.parametrize("act_size", (2,))
    @pytest.mark.parametrize("planning_steps", (3, 5))
    @pytest.mark.parametrize("num_rollouts", (1, 4))
    def test_step(self, state_size, act_size, planning_steps, num_rollouts):
        generative_env = MockGenerativeEnv(state_size, act_size)

        def action_sampler(obs, steps):
            return torch.rand((num_rollouts, steps, act_size))

        discount_factor = 0.5
        dut = RandomShootingAgent(
            generative_env, action_sampler, planning_steps, discount_factor
        )
        seed = 1234
        torch.manual_seed(seed)

        obs = torch.ones((state_size,))
        act_sequence, state_sequence, best_reward = dut.step(obs)

        assert act_sequence.shape == (planning_steps, act_size)
        assert state_sequence.shape == (planning_steps + 1, state_size)

        # Make sure that state_sequence, best_reward is consistent with act_sequence.
        reward_expected = 0
        np.testing.assert_allclose(
            state_sequence[0].detach().numpy(),
            generative_env.dynamics_model.state_from_observation(obs).detach().numpy(),
        )
        for i in range(planning_steps):
            next_state, _, _ = generative_env.dynamics_model.step(
                state_sequence[i], act_sequence[i]
            )
            obs_i = generative_env.dynamics_model.observation(state_sequence[i])
            np.testing.assert_allclose(
                state_sequence[i + 1].detach().numpy(),
                next_state.detach().numpy(),
                atol=1e-6,
            )
            reward, _ = generative_env.reward_model.step(
                state_sequence[i], obs_i, act_sequence[i]
            )
            reward_expected += reward * discount_factor**i
        np.testing.assert_allclose(best_reward.item(), reward_expected.item())
