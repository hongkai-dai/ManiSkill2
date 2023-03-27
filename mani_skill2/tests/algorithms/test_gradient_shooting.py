import numpy as np
import pytest
import torch

from mani_skill2.algorithms.gradient_shooting import GradientShootingAgent
from mani_skill2.utils.testing.dynamics_test_utils import MockGenerativeEnv


class TestGradientShootingAgent:
    @pytest.mark.parametrize("state_size", (2,))
    @pytest.mark.parametrize("act_size", (3,))
    @pytest.mark.parametrize("planning_steps", (3,))
    @pytest.mark.parametrize("batch_size", (1, 3))
    @pytest.mark.parametrize("device", {"cpu", "cuda"})
    def test_compute_rollout_gradient(
        self, state_size, act_size, planning_steps, batch_size, device
    ):
        """
        Test the gradient computation in step() function.
        """
        torch.random.manual_seed(123)
        generative_env = MockGenerativeEnv(state_size, act_size)
        generative_env.dynamics_model.to(device)
        generative_env.reward_model.to(device)

        discount_factor = 0.5
        gradient_steps = 1
        action_bounds = (
            -torch.ones((act_size,), device=device),
            torch.ones((act_size,), device=device),
        )
        dut = GradientShootingAgent(
            generative_env,
            planning_steps,
            discount_factor,
            torch.optim.Adam,
            gradient_steps,
            action_bounds,
            device,
        )

        initial_states = torch.rand(
            (
                batch_size,
                state_size,
            ),
            device=device,
        )
        act_sequences = torch.rand(
            (batch_size, planning_steps, act_size), device=device
        )
        act_sequences.requires_grad_(True)

        best_rewards = torch.full((batch_size,), -np.inf, device=device)
        best_act_sequences = torch.empty(
            (batch_size, planning_steps, act_size), device=device
        )

        info = dut._compute_rollout_gradient(
            initial_states, act_sequences, best_rewards, best_act_sequences
        )
        assert isinstance(info, dict)
        assert not torch.isinf(best_rewards).any()

        act_sequences_grad = act_sequences.grad.clone()

        # Now compute the gradient for each initial state in a for loop
        for i in range(batch_size):
            act_sequences.grad.zero_()
            _, total_reward, _ = dut.generative_env.rollout(
                initial_states[i].unsqueeze(0),
                act_sequences[i].unsqueeze(0),
                discount_factor,
            )
            total_loss = -total_reward
            total_loss.backward()
            np.testing.assert_allclose(
                act_sequences_grad[i].cpu().detach().numpy(),
                act_sequences.grad[i].cpu().detach().numpy(),
                atol=1e-5,
            )

        # Now compute the total reward using another random action sequences. Check if best_rewards and best_act_sequences are updated.
        best_rewards_old = best_rewards.clone()
        best_act_sequences_old = best_act_sequences.clone()
        new_act_sequences = torch.rand_like(act_sequences, device=device)
        new_act_sequences.requires_grad_(True)
        dut._compute_rollout_gradient(
            initial_states, new_act_sequences, best_rewards, best_act_sequences
        )
        for i in range(batch_size):
            _, total_reward, _ = dut.generative_env.rollout(
                initial_states[i].unsqueeze(0),
                new_act_sequences[i].unsqueeze(0),
                discount_factor,
            )
            if total_reward > best_rewards_old[i]:
                np.testing.assert_allclose(total_reward.item(), best_rewards[i].item())
                np.testing.assert_allclose(
                    new_act_sequences[i].cpu().detach().numpy(),
                    best_act_sequences[i].cpu().detach().numpy(),
                )
            else:
                np.testing.assert_allclose(
                    best_rewards_old[i].item(), best_rewards[i].item()
                )
                np.testing.assert_allclose(
                    best_act_sequences[i].cpu().detach().numpy(),
                    best_act_sequences_old[i].cpu().detach().numpy(),
                )

    @pytest.mark.parametrize("state_size", (3,))
    @pytest.mark.parametrize("act_size", (2,))
    @pytest.mark.parametrize("planning_steps", (3, 5))
    @pytest.mark.parametrize("gradient_steps", (1, 5))
    @pytest.mark.parametrize("batch_size", (1, 3))
    @pytest.mark.parametrize("device", {"cpu", "cuda"})
    def test_step(
        self, state_size, act_size, planning_steps, gradient_steps, batch_size, device
    ):
        torch.random.manual_seed(123)
        generative_env = MockGenerativeEnv(state_size, act_size)
        generative_env.dynamics_model.to(device)
        generative_env.reward_model.to(device)

        discount_factor = 0.5
        action_bounds = (
            -torch.ones((act_size,), device=device),
            torch.ones((act_size,), device=device),
        )
        dut = GradientShootingAgent(
            generative_env,
            planning_steps,
            discount_factor,
            torch.optim.Adam,
            gradient_steps,
            action_bounds,
            device,
        )

        obs = np.ones((state_size,))
        act_sequences_init = torch.zeros(
            (batch_size, planning_steps, act_size), device=device
        )
        with torch.no_grad():
            state_init = generative_env.dynamics_model.state_from_observation(obs).to(
                device
            )
            state_init_repeat = state_init.repeat([batch_size] + [1] * state_init.ndim)
            state_sequences_init, reward_init, _ = generative_env.rollout(
                state_init_repeat, act_sequences_init, discount_factor
            )

        dut.set_action_sequences_init(act_sequences_init)
        best_next_action, info = dut.step(obs)
        assert best_next_action.shape == (act_size,)
        assert isinstance(best_next_action, np.ndarray)
        assert isinstance(info, dict)
        best_rewards = info["best_rewards"]
        np.testing.assert_array_equal(
            info["best_act_sequences"][torch.argmax(best_rewards), 0, ...]
            .cpu()
            .detach()
            .numpy(),
            best_next_action,
        )
        with torch.no_grad():
            states_optimized, reward_optimized, _ = generative_env.rollout(
                state_init_repeat, info["best_act_sequences"], discount_factor
            )
            assert reward_optimized[0].item() >= reward_init[0].item()
