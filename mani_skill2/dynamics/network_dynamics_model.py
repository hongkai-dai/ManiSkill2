from typing import Dict, Tuple

import torch
import torch.nn as nn

from mani_skill2.dynamics.base import DynamicsModel


class NetworkDynamicsModel(DynamicsModel):
    """A dynamics model that uses an underlying network."""

    reset_state: torch.Tensor

    def __init__(self, network: nn.Module, is_residual: bool):
        """
        Args:
            network: The network used to generate the next state or a residual of it.
            is_residual: If True, the network output is interpreted as a residual.
        """
        super().__init__()
        self.network = network
        self.is_residual = is_residual

    def step(
        self, state: torch.Tensor, act: torch.Tensor
    ) -> Tuple[torch.Tensor, bool, Dict]:
        """See base class."""
        if self.is_residual:
            pred_residual = self.network(state, act)
            pred_next_state = state + pred_residual
        else:
            pred_next_state = self.network(state, act)
            pred_residual = pred_next_state - state

        info = dict(pred_residual=pred_residual)
        return pred_next_state, False, info

    def set_reset(self, reset_state: torch.Tensor):
        self.reset_state = reset_state

    def reset(self) -> torch.Tensor:
        return self.reset_state
