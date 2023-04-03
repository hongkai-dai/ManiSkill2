import abc
from typing import Optional

import torch


class Normalizer(abc.ABC):
    """Normalizes / denormalizes values."""

    @abc.abstractmethod
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes the value."""

    @abc.abstractmethod
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalizes the value."""


class HeightMapNormalizer(Normalizer):
    """Normalizes height maps by scaling them, and clamp at zero."""

    def __init__(self, scale: float = 10.0, heightmap_min: Optional[float] = 0.0):
        """
        Args:
            heightmap_min: Clamp the heightmap before scaling at this minimal value.
        """
        assert scale > 0
        self.scale = scale
        self.heightmap_min = heightmap_min

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.heightmap_min is not None:
            x_clamped = torch.clamp(x, min=self.heightmap_min)
        else:
            x_clamped = x
        return x_clamped * self.scale

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        x_descaled = x / self.scale
        if self.heightmap_min is not None:
            return torch.clamp(x_descaled, min=self.heightmap_min)
        else:
            return x_descaled


class AffineNormalizer(Normalizer):
    """
    We estimate the input bounds to be within [lo, up], and we apply an affine
    transformation 2 * (input - lo) / (up - lo) - 1 which will scale it to the range
    [-1, 1]
    """

    def __init__(self, lo: torch.Tensor, up: torch.Tensor):
        assert lo.shape == up.shape
        assert torch.all(lo <= up)
        self.lo = lo
        self.up = up
        # If self.lo[i] = self.up[i], don't scale.
        self.scale_flag = self.lo != self.up

    @classmethod
    def from_data(cls, data: torch.Tensor):
        """
        Use the lower and upper value of a batch of data to construct the normalizer.

        Args:
            data: Size is data_batch_size * per_data_size. We will find the maximum and
            minimum along dim=0 as lo and up
        """
        return cls(torch.min(data, dim=0).values, torch.max(data, dim=0).values)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A batch of data, x is of shape (batch_size, per_data_size)
        """
        if x.shape == self.lo.shape:
            # Single data
            return self.normalize(x.unsqueeze(0)).squeeze(0)
        else:
            # batch of data
            x_normalized = x.clone()
            device = x.device
            x_normalized[:, self.scale_flag] = (
                2
                * (x_normalized[:, self.scale_flag] - self.lo[self.scale_flag].to(device))
                / (self.up[self.scale_flag].to(device) - self.lo[self.scale_flag].to(device))
                - 1
            )
            return x_normalized

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A batch of data, x is of shape (batch_size, per_data_size)
        """
        if x.shape == self.lo.shape:
            # Single data
            return self.denormalize(x.unsqueeze(0)).squeeze(0)
        # A batch of data
        device = x.device
        x_denormalize = x.clone()
        x_denormalize[:, self.scale_flag] = (x_denormalize[:, self.scale_flag] + 1) * (
            self.up[self.scale_flag].to(device) - self.lo[self.scale_flag].to(device)
        ) / 2 + self.lo[self.scale_flag].to(device)
        return x_denormalize
