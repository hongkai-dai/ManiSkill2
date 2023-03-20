from typing import Tuple, Dict

import numpy as np
import torch

from mani_skill2.dynamics.reward import GoalBasedRewardModel


# TODO(blake.wulfe): Where to put this?
class FlatDoughRollingRewardModel(GoalBasedRewardModel):
    def step(
        self, state: torch.Tensor, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[float, Dict]:
        dims = tuple(range(1, state.ndim))
        num_pos = (state > 0).sum(dim=dims)
        avg_pos = state.sum(dim=dims) / (num_pos + 1e-12)
        rew = -avg_pos * 10
        return rew, {}

    def set_goal(self, goal):
        pass


class EllipseShapeRewardModel(GoalBasedRewardModel):
    """
    For a planar ellipse shape with uniform height, use the negative of the heightmap MSE error as the reward.
    """

    def __init__(
        self,
        total_volume: float,
        length: float,
        height: float,
        grid_coord_h: torch.Tensor,
        grid_coord_w: torch.Tensor,
    ):
        """

        Args:
          grid_coord_h: A 1-D tensor. The grid coordinate along the image H dimension
          grid_coord_w: A 1-D tensor. The grid coordinate along the image W dimension
        """
        self.height = height
        self.length = length
        area = total_volume / height
        self.width = area / (np.pi * self.length)
        # We will sample many orientation angle and position of the ellipsoid to shift the ellipsoid.
        self.angle_samples = torch.tensor([0.0])
        self.position_samples = torch.tensor([[0.0, 0.0]])
        self.grid_coord_h = grid_coord_h
        self.grid_coord_w = grid_coord_w

    def desired_heightmap(
        self, center_position: torch.Tensor, angle: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate the heightmap of the ellipsoid by shifting its pose.

        Args:
          center_position: The center of the ellipsoid in the heightmap grid.
          angle: the orientation angle of the ellipsoid.

        Returns:
          height: height[i, j] is the height in the heightmap at (grid_coord_h[i], grid_coord_w[j])
        """
        grid_h, grid_w = torch.meshgrid(
            self.grid_coord_h, self.grid_coord_w, indexing="ij"
        )
        # We first compute the grid coordinate, and then convert that grid to the
        # ellipse frame. We can easily determine if a grid point is within the ellipse
        # in the ellipse frame.

        # Each row of grid_coord is a grid point.
        grid_coord = torch.cat((grid_h.unsqueeze(2), grid_w.unsqueeze(2)), dim=-1).view(
            -1, 2
        )
        cos_theta = torch.cos(angle)
        sin_theta = torch.sin(angle)
        R = torch.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        grid_coord_ellipse = (grid_coord - center_position) @ R
        in_ellipse_mask = (
            grid_coord_ellipse[:, 0] ** 2 / self.length**2
            + grid_coord_ellipse[:, 1] ** 2 / self.width**2
        ) <= 1
        height = self.height * in_ellipse_mask.view(grid_h.shape)
        return height

    def set_goal(self, goal: torch.Tensor):
        pass

    def step(
        self, state: torch.Tensor, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute the reward for a batch of (state, obs, action) tuples.
        We use the negation of the heightmap MSE and l1-norm error as the reward.

        Args:
          state: Size is (batch_size, state_size) This is a batch of height maps.
          obs: Size is (batch_size, obs_size) We will not use obs
          action: Size is (batch_size, act_size)

        Return:
          reward Size is (batch_size) reward[i] is the reward for state[i], obs[i], action[i]
          info:
        """
        # For the moment, assume that we only have one single orientation and position sample.
        assert self.angle_samples.numel() == 1
        assert self.position_samples.shape[0] == 1
        batch_size = state.shape[0]
        desired_height = self.desired_heightmap(
            self.position_samples[0], self.angle_samples[0]
        )

        desired_height_repeated = desired_height.repeat(
            [batch_size] + [1] * desired_height.ndim
        )
        # A combination of MSE and L1 loss.
        diff = (state - desired_height_repeated).view((state.shape[0], -1))
        error = (diff**2).mean(dim=-1) + torch.abs(diff).mean(dim=-1)
        return -error, dict()
