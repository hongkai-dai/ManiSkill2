import abc

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
    """Normalizes height maps by scaling them."""

    def __init__(self, scale: float = 10.0):
        assert scale > 0
        self.scale = scale

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.scale
