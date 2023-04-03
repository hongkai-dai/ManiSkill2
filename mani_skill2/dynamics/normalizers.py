import abc
import math
import numpy as np
from typing import List, Optional, Tuple

import torch


class Normalizer(abc.ABC, torch.nn.Module):
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
        super().__init__()
        assert scale > 0
        self.scale = scale
        self.heightmap_min = heightmap_min

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.heightmap_min is not None:
            return torch.clamp(x, min=self.heightmap_min) * self.scale
        else:
            return x * self.scale

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

    def __init__(self, data_shape: Tuple[int], lo: torch.Tensor, up: torch.Tensor):
        super().__init__()
        if lo is not None and up is not None:
            assert lo.shape == up.shape
            assert torch.all(lo <= up)
        if lo is None:
            # Do not register self.lo with value None, it would not count self.lo in
            # the state_dict then.
            lo = torch.full(tuple(data_shape), -np.inf)
        else:
            assert lo.shape == tuple(data_shape)
        if up is None:
            up = torch.full(tuple(data_shape), np.inf)
        else:
            assert up.shape == tuple(data_shape)

        self.register_buffer("lo", lo)
        self.register_buffer("up", up)
        self.register_buffer("scale_flag", None)
        if self.lo is not None and self.up is not None:
            # If self.lo[i] = self.up[i], don't scale.
            self.scale_flag = self.lo != self.up

    @classmethod
    def from_data(cls, data: torch.Tensor, percentile: float):
        """
        Use the lower and upper value of a batch of data to construct the normalizer.

        Args:
            data: Size is data_batch_size * per_data_size. We will find the maximum and
            minimum along dim=0 as lo and up
            percentile: We take min / max as percentile and (1-percentile) of data.
        """
        assert percentile >= 0
        # Sort in ascending order.
        _, indices = torch.sort(data, dim=0)
        column_indices = torch.arange(data.shape[1])
        lo = data[indices[math.floor(data.shape[0] * percentile)], column_indices]
        up = data[
            indices[math.ceil(data.shape[0] * (1 - percentile)) - 1], column_indices
        ]
        return cls(list(lo.shape), lo, up)

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
            x_normalized[:, self.scale_flag] = (
                2
                * (x_normalized[:, self.scale_flag].clone() - self.lo[self.scale_flag])
                / (self.up[self.scale_flag] - self.lo[self.scale_flag])
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
        x_denormalize = x.clone()
        x_denormalize[:, self.scale_flag] = (
            x_denormalize[:, self.scale_flag].clone() + 1
        ) * (self.up[self.scale_flag] - self.lo[self.scale_flag]) / 2 + self.lo[
            self.scale_flag
        ]
        return x_denormalize
