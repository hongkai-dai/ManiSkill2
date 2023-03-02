from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from mani_skill2.dynamics.base import DynamicsModel
from mani_skill2.dynamics.normalizers import Normalizer


class NetworkDynamicsModel(DynamicsModel):
    """A dynamics model that uses an underlying network."""

    reset_state: torch.Tensor

    def __init__(
        self,
        network: nn.Module,
        is_residual: bool,
        state_normalizer: Optional[Normalizer] = None,
        action_normalizer: Optional[Normalizer] = None,
        output_normalizer: Optional[Normalizer] = None,
    ):
        """
        Args:
            network: The network used to generate the next state or a residual of it.
            is_residual: If True, the network output is interpreted as a residual.
            state_normalizer: Normalizes the state before inputting to network.
            action_normalizer: Normalizes the action before inputting to network.
            output_normalizer: Denormalizes the output before returning it.
        """
        super().__init__()
        self.network = network
        self.is_residual = is_residual
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        self.output_normalizer = output_normalizer

    def step(
        self,
        state: torch.Tensor,
        act: torch.Tensor,
    ) -> Tuple[torch.Tensor, bool, Dict]:
        """See base class.

        This normalizes the input (state and action) and denormalizes the output.
        The reason it denormalizes the output is to ensure that the output exists
        in the space expected by planning algorithms, which should match the
        original observation space of the environment.

        The assumption is that during training the training module normalizes
        the target (gt) next state (_and_ pred_next_state) iff the output_normalizer
        exists on this class.
        """
        if self.state_normalizer is not None:
            state = self.state_normalizer.normalize(state)
        if self.action_normalizer is not None:
            action = self.action_normalizer.normalize(action)

        if self.is_residual:
            pred_residual = self.network(state, act)
            pred_next_state = state + pred_residual
        else:
            pred_next_state = self.network(state, act)
            pred_residual = pred_next_state - state

        if self.output_normalizer is not None:
            pred_next_state = self.output_normalizer.denormalize(pred_next_state)
            pred_residual = self.output_normalizer.denormalize(pred_residual)

        info = dict(pred_residual=pred_residual)
        return pred_next_state, False, info

    def set_reset(self, reset_state: torch.Tensor):
        self.reset_state = reset_state

    def reset(self) -> torch.Tensor:
        return self.reset_state
