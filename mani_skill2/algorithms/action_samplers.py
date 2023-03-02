from typing import Tuple

import numpy as np
import torch


class ConstantValue:
    """Returns a constant value in response to sample().

    This allows for using a consistent interface for sampling values
    even when the value being sampled is constant.
    """

    def __init__(self, value: float):
        self.value = value

    def sample(self, *args, **kwargs) -> float:
        return self.value


def _validate_bounds(bounds: Tuple[float, float]) -> None:
    assert len(bounds) == 2
    lb, ub = bounds
    assert lb <= ub


def _get_uniform_dist_or_constant_from_bounds(bounds: Tuple[float, float]):
    _validate_bounds(bounds)
    lb, ub = bounds
    if lb == ub:
        return ConstantValue(lb)
    else:
        return torch.distributions.Uniform(lb, ub)


# TODO(blake.wulfe): Probably have one of these per type of dough rolling actions.
# TODO(blake.wulfe): Add an ABC for this if it makes sense.
class RandomDoughRollingActionSampler:
    """Samples actions for dough rolling with random values between bounds.

    TODO(blake.wulfe): Add bounds for other values if necessary.
    """

    def __init__(
        self,
        action_size: int,
        num_samples: int = 100,
        height_bounds: Tuple[float, float] = (0.04, 0.06),
        rolling_distance_bounds: Tuple[float, float] = (0.1, 0.1),
        rolling_duration_bounds: Tuple[float, float] = (1.0, 1.0),
        device: str = "cuda",
    ):
        """
        Args:
            action_size: Size of the action. This is assumed to be 10 for now.
            num_samples: Number of action samples to return.
            height_bounds: Tuple of lower and upper bounds to sample values.
            rolling_distance_bounds: Distance to roll bounds.
            rolling_duration_bounds: Time to take in performing the roll bounds.
            device: Device on which to generate the actions.
        """
        self.action_size = action_size
        self.num_samples = num_samples
        self.height_distribution = _get_uniform_dist_or_constant_from_bounds(
            height_bounds,
        )
        self.rolling_distance_distribution = _get_uniform_dist_or_constant_from_bounds(
            rolling_distance_bounds,
        )
        self.rolling_duration_distribution = _get_uniform_dist_or_constant_from_bounds(
            rolling_duration_bounds,
        )
        self.device = device

    def __call__(self, obs: np.ndarray, num_steps: int) -> torch.Tensor:
        """
        Args:
            obs: The observation to generate actions for. Assumed
                to be a numpy array.
            num_steps: Horizon of actions to generate.

        Returns:
            A tensor of shape (num_samples, num_steps, action_size).
        """
        actions = torch.zeros(
            (self.num_samples, num_steps, self.action_size),
            device=self.device,
        )
        # Duration.
        actions[..., 0] = self.rolling_duration_distribution.sample(
            (self.num_samples, num_steps),
        )
        # Height.
        actions[..., 3] = self.height_distribution.sample(
            (self.num_samples, num_steps),
        )
        # Yaw.
        actions[..., 4] = torch.rand((self.num_samples, num_steps)) * 2 * torch.pi
        # Distance.
        actions[..., 6] = self.rolling_distance_distribution.sample(
            (self.num_samples, num_steps),
        )
        return actions
