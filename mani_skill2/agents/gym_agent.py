import abc

class GymAgent(abc.ABC):
    """An agent in the Gym sense (as opposed to Sapien sense)."""

    def reset(self):
        pass

    @abc.abstractmethod
    def step(self, obs):
        """Returns the action of the agent at this timestep."""
        raise NotImplementedError

