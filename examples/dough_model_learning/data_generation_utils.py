import abc
from dataclasses import dataclass
import typing

import numpy as np
import torch
import torch.utils.data

import mani_skill2.envs.mpm.rolling_env as rolling_env


@dataclass
class SampleActionOptions:
    position_std: float = 0.1
    height_std: float = 0.03


def sample_action(
    env: rolling_env.RollingEnv,
    options: SampleActionOptions,
) -> np.ndarray:
    """Generate a sampled action for the RollingPin environment."""
    while True:
        if env.action_option == rolling_env.ActionOption.LIFTAFTERROLL:
            duration = 1.0
            # Sample a starting position not too far away from the dough center.
            start_position = env.dough_center()
            start_position[:2] += np.random.randn(2) * options.position_std
            start_position[2] += np.random.randn() * options.height_std
            # The initial yaw angle is arbitrary.
            start_yaw = np.random.rand() * np.pi * 2
            # The initial pitch angle is some small angle.
            start_pitch = (np.random.rand() - 0.5) * 0.1 * np.pi
            rolling_distance = (np.random.rand() - 0.5) * 0.4
            delta_height = (np.random.rand() - 0.5) * 0.05
            delta_yaw = (np.random.rand() - 0.5) * 0.2 * np.pi
            delta_pitch = (np.random.rand() - 0.5) * 0.1 * np.pi
            action = np.concatenate(
                (
                    [duration],
                    start_position,
                    [
                        start_yaw,
                        start_pitch,
                        rolling_distance,
                        delta_height,
                        delta_yaw,
                        delta_pitch,
                    ],
                )
            )
            if env.is_action_valid(action):
                return action
        else:
            raise NotImplementedError


@dataclass
class TransitionTuple:
    current_state: rolling_env.Heightmap
    action: np.ndarray
    next_state: rolling_env.Heightmap


@dataclass
class GenerateTransitionTupleOptions:
    sample_action_options: SampleActionOptions = SampleActionOptions()
    grid_size: typing.Tuple[int] = (64, 64)
    dx: float = 0.01


def generate_transition_tuple(
    env: rolling_env.RollingEnv,
    options: GenerateTransitionTupleOptions,
) -> TransitionTuple:
    """Generates a single transition from an initial dough configuration.

    For the current dough rolling environemnt, sample an action, and then
    generate the tuple (current_state, action, next_state).
    """
    current_state = env.calc_heightmap(options.dx, options.grid_size)
    action = sample_action(env, options.sample_action_options)
    env.step(action)
    next_state = env.calc_heightmap(options.dx, options.grid_size)
    return TransitionTuple(current_state, action, next_state)


class TransitionTupleDataset(torch.utils.data.TensorDataset):
    # I will assume all heightmaps are generated on the same grid.
    grid_h: torch.Tensor
    grid_w: torch.Tensor

    def __init__(
        self,
        transition_tuples: typing.List[TransitionTuple],
        dtype=torch.float64,
    ):
        self.grid_h = torch.from_numpy(transition_tuples[0].current_state.grid_h).to(
            dtype
        )
        self.grid_w = torch.from_numpy(transition_tuples[0].current_state.grid_w).to(
            dtype
        )

        current_heights = torch.empty(
            (len(transition_tuples), *transition_tuples[0].current_state.height.shape),
            dtype=dtype,
        )
        actions = torch.empty(
            (len(transition_tuples), *transition_tuples[0].action.shape), dtype=dtype
        )
        next_heights = torch.empty(
            (len(transition_tuples), *transition_tuples[0].next_state.height.shape),
            dtype=dtype,
        )
        for i, transition_tuple in enumerate(transition_tuples):
            # Check if the heightmaps are generated on the same grid.
            assert np.array_equal(
                transition_tuple.current_state.grid_w,
                transition_tuples[0].current_state.grid_w,
            )
            assert np.array_equal(
                transition_tuple.current_state.grid_h,
                transition_tuples[0].current_state.grid_h,
            )
            assert np.array_equal(
                transition_tuple.next_state.grid_w,
                transition_tuples[0].current_state.grid_w,
            )
            assert np.array_equal(
                transition_tuple.next_state.grid_h,
                transition_tuples[0].current_state.grid_h,
            )
            current_heights[i] = torch.from_numpy(
                transition_tuple.current_state.height
            ).to(dtype)
            actions[i] = torch.from_numpy(transition_tuple.action).to(dtype)
            next_heights[i] = torch.from_numpy(transition_tuple.next_state.height).to(
                dtype
            )
        super(TransitionTupleDataset, self).__init__(
            current_heights, actions, next_heights
        )


class GymAgent(abc.ABC):
    """An agent in the Gym sense (as opposed to Sapien sense)."""

    def reset(self):
        pass

    @abc.abstractmethod
    def step(self, obs):
        """Returns the action of the agent at this timestep."""
        raise NotImplementedError


class DoughRollingCenterOutAgent(GymAgent):
    """Rolls dough starting from the center and moving outward."""

    def __init__(
        self,
        duration: float = 1.0,
        start_height: float = 0.06,
        height_delta_between_sweeps: float = 0.0025,
    ):
        """
        Args:
            duration: Time taken to perform the rolling action.
            start_height: Height to start rolling pin at the beginning of the episode.
            height_delta_between_sweeps: Change in height between sweeps.
                Note this is not the change in height _during_ a timestep.
        """
        self.duration = duration
        self.start_height = start_height
        self.height_delta_between_sweeps = height_delta_between_sweeps

    def reset(self):
        self.height = self.start_height

    def step(self, obs):
        height = self.height
        self.height -= self.height_delta_between_sweeps

        start_position = [0, 0, height]
        start_yaw = np.random.uniform(0, 2 * np.pi)
        start_pitch = 0.0
        rolling_distance = 0.1
        delta_height = 0.0
        delta_yaw = 0.0
        delta_pitch = 0.0
        action = np.concatenate(
            (
                [self.duration],
                start_position,
                [
                    start_yaw,
                    start_pitch,
                    rolling_distance,
                    delta_height,
                    delta_yaw,
                    delta_pitch,
                ],
            )
        )
        return action
