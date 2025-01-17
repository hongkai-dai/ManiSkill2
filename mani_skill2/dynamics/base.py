import abc
from typing import Dict, List, Tuple, Union

import numpy as np
import torch


class DynamicsModel(abc.ABC, torch.nn.Module):
    """Abstract base class for a dynamics model."""

    @abc.abstractmethod
    def step(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, bool, Dict]:
        """Steps the dynamics forward one transition.

        Args:
            state: The state at the beginning of this transition.
            action: The action at this transition.

        Returns:
            next_state (torch.Tensor): the state at the end of this transition.
            terminated (bool): Whether this transition should be terminated.
            info (Dict): auxiliary diagnostic information
        """

    @abc.abstractmethod
    def reset(self) -> torch.Tensor:
        """Resets any internal state; called at the beginning of an episode."""

    def observation(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes the observation for a given state.
        The default behavior is to assume fully-observable state, and return the state
        as the observation.
        """
        return state

    def state_from_observation(self, obs: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Computes (estimates) the state from an observation.

        The default behavior is to assume fully-observable state, and return the
        observation as the state.
        """
        if isinstance(obs, np.ndarray):
            return torch.from_numpy(obs)
        else:
            return obs
