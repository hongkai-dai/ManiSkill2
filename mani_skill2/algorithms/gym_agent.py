import abc
from typing import Any, Tuple, Dict


class GymAgent(abc.ABC):
    """An agent in the Gym sense (as opposed to Sapien sense)."""

    def reset(self):
        pass

    @abc.abstractmethod
    def step(self, obs: Any) -> Tuple[Any, Dict]:
        """Computes the action of the agent at this timestep.

        Args:
            obs: The observation from the environment.

        Returns:
            A tuple of (action, info), where the action should
            be applied to the env and the info is internal information
            that can vary per subclass.
        """
        raise NotImplementedError
