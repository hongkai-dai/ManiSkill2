import numpy as np
import torch

# TODO(blake.wulfe): Probably have one of these per type of dough rolling actions.
# TODO(blake.wulfe): Add a ABC for this if it makes sense.
class RandomDoughRollingActionSampler:
    """Samples actions for dough rolling with random height and yaw.

    TODO(blake.wulfe): extend this to sample the other action components.
    """

    def __init__(
        self,
        action_size: int,
        num_samples: int = 100,
        height_lb: float = 0.04,
        height_ub: float = 0.06,
        rolling_distance: float = 0.1,
        rolling_duration: float = 1.0,
        device: str = "cuda",
    ):
        """
        Args:
            action_size: Size of the action. This is assumed to be 10 for now.
            num_samples: Number of action samples to return.
            height_lb: Lower bound of the height to sample.
            height_ub: Upper bound of the height to sample.
            rolling_distance: Distance to roll.
            rolling_duration: Time to take in performing the roll.
            device: Device on which to generate the actions.
        """
        self.action_size = action_size
        self.num_samples = num_samples
        assert height_lb < height_ub
        self.height_distribution = torch.distributions.Uniform(height_lb, height_ub)
        self.rolling_distance = rolling_distance
        self.rolling_duration = rolling_duration
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
        actions[..., 0] = self.rolling_duration
        # Height.
        actions[..., 3] = self.height_distribution.sample((self.num_samples, num_steps))
        # Yaw.
        actions[..., 4] = torch.rand((self.num_samples, num_steps)) * 2 * torch.pi
        # Distance.
        actions[..., 6] = self.rolling_distance
        return actions
