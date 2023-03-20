import numpy as np
import torch
import pytest

import mani_skill2.dough_model_learning.dough_reward_models as mut


class TestEllipseShapeRewardModel:
    @pytest.mark.parametrize("angle", [torch.tensor(0.0), torch.tensor(1.1)])
    @pytest.mark.parametrize(
        "position", (torch.tensor([0.0, 0.0]), torch.tensor([0.2, 0.05]))
    )
    def test_desired_heightmap(self, angle, position):
        dut = mut.EllipseShapeRewardModel(
            total_volume=0.01,
            length=0.5,
            height=0.1,
            grid_coord_h=torch.linspace(-0.5, 0.5, 20),
            grid_coord_w=torch.linspace(-1, 1, 30),
        )
        desired_height = dut.desired_heightmap(position, angle)
        assert desired_height.shape == (
            dut.grid_coord_h.numel(),
            dut.grid_coord_w.numel(),
        )
        cos_theta = torch.cos(angle)
        sin_theta = torch.sin(angle)
        R = torch.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        # Now I sample many points and check the height.
        for i in range(dut.grid_coord_h.numel()):
            for j in range(dut.grid_coord_w.numel()):
                coord = torch.stack((dut.grid_coord_h[i], dut.grid_coord_w[j]))
                coord_ellipse = R.T @ (coord - position)
                height_expected = 0
                if (
                    coord_ellipse[0] ** 2 / dut.length**2
                    + coord_ellipse[1] ** 2 / dut.width**2
                    <= 1
                ):
                    height_expected = dut.height
                np.testing.assert_allclose(desired_height[i, j].item(), height_expected)

    def test_step(self):
        dut = mut.EllipseShapeRewardModel(
            total_volume=0.01,
            length=0.5,
            height=0.1,
            grid_coord_h=torch.linspace(-0.5, 0.5, 20),
            grid_coord_w=torch.linspace(-1, 1, 30),
        )
        batch_size = 2
        state = torch.ones(
            (batch_size, dut.grid_coord_h.numel(), dut.grid_coord_w.numel())
        )
        state[batch_size - 1] *= 2
        obs = torch.ones(
            (batch_size, dut.grid_coord_h.numel(), dut.grid_coord_w.numel())
        )
        # Random action size. We don't compute the cost on the action.
        act_size = 10
        action = torch.zeros((batch_size, act_size))
        reward, info = dut.step(state, obs, action)
        assert reward.shape == (batch_size,)
        assert isinstance(info, dict)
        desired_height = dut.desired_heightmap(
            dut.position_samples[0], dut.angle_samples[0]
        )
        for i in range(batch_size):
            error_expected = torch.nn.functional.mse_loss(
                state[i], desired_height
            ) + torch.nn.functional.l1_loss(state[i], desired_height)
            np.testing.assert_allclose(reward[i].item(), -error_expected.item())
