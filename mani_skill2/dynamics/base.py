import abc
from typing import Dict, Tuple, TypeVar

import torch

StateType = TypeVar("StateType")
ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")


class DynamicsModel(abc.ABC, torch.nn.Module):
    """Abstract base class for a dynamics model."""

    @abc.abstractmethod
    def step(self, state: StateType, act: torch.Tensor) -> Tuple[StateType, bool, Dict]:
        """Steps the dynamics forward one transition.

        Args:
            state: The state at the beginning of this transition.
            act: The action at this transition.

        Returns:
            next_state (StateType): the state at the end of this transition.
            terminated (bool): Whether this transition should be terminated.
            info (Dict): auxiliary diagnostic information
        """

    @abc.abstractmethod
    def reset(self) -> StateType:
        """Resets any internal state; called at the beginning of an episode."""

    def observation(self, state: StateType) -> ObsType:
        """
        Computes the observation for a given state.
        The default behavior is to assume fully-observable state, and return the state
        as the observation.
        """
        return state
