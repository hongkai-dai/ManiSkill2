import pytest
import torch

from mani_skill2.algorithms.gradient_shooting import GradientShootingAgent
from mani_skill2.utils.testing.dynamics_test_utils import MockGenerativeEnv


class TestGradientShootingAgent:
    @pytest.mark.parametrize("state_size", (3,))
    @pytest.mark.parametrize("act_size", (2,))
    @pytest.mark.parametrize("planning_steps", (3, 5))
    @pytest.mark.parametrize("gradient_steps", (2, 5))
    def test_step(self, state_size, act_size, planning_steps, gradient_steps):
        device = "cuda"
        generative_env = MockGenerativeEnv(state_size, act_size)
        generative_env.dynamics_model.to(device)
        generative_env.reward_model.to(device)

        discount_factor = 0.5
        dut = GradientShootingAgent(
            generative_env,
            planning_steps,
            discount_factor,
            torch.optim.Adam,
            gradient_steps,
        )

        obs = torch.ones((state_size,), device=device)
        act_sequence_init = torch.empty((planning_steps, act_size), device=device)
        with torch.no_grad():
            state_init = generative_env.dynamics_model.state_from_observation(obs)
            states, reward_init = generative_env.rollout(
                state_init.unsqueeze(0), act_sequence_init.unsqueeze(0), discount_factor
            )
            print(reward_init)

        dut.set_action_sequence_init(act_sequence_init)
        act_sequence = dut.step(obs)
        assert act_sequence.shape == (planning_steps, act_size)
        with torch.no_grad():
            _, reward_optimized = generative_env.rollout(
                state_init.unsqueeze(0), act_sequence.unsqueeze(0), discount_factor
            )
            assert reward_optimized[0].item() > reward_init[0].item()
