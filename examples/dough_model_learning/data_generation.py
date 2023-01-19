from dataclasses import dataclass
import typing

import numpy as np
import torch
import torch.utils.data

import mani_skill2.envs.mpm.rolling_env as rolling_env


def generate_circular_cone_heightmap(radius: float, height: float,
                                     dx: float) -> np.ndarray:
    half_width = int(radius / dx)
    width = 2 * half_width + 1
    height_map = np.zeros((width, width), dtype=np.float32)
    X, Y = np.meshgrid(np.linspace(-half_width * dx, half_width * dx, width),
                       np.linspace(-half_width * dx, half_width * dx, width))
    height_map = height - np.sqrt(X**2 + Y**2) / radius * height
    height_map = np.clip(height_map,
                         a_min=np.zeros_like(height_map),
                         a_max=None)
    return height_map


def generate_dome_heightmap(dome_radius: float, dome_height: float,
                            dx: float) -> np.ndarray:
    """
    Generate the initial heightmap as a dome.

    I assume the initial heightmap is a dome, obtained by rotating an arc about
    its symmetric axis, which also aligns with the vertical axis in the world.

    Args:
      dome_radius: Projecting this dome to the horizontal plane, we get a
      circle. dome_radius is the radius of this projected circle.
      dome_height: The height from the top of the dome to the bottom of the dome.
    """

    # This dome is a part of a sphere. We compute the sphere radius.
    # By Pythagorean theorem, we have
    # (sphere_radius - dome_height)² + dome_radius² = sphere_radius²
    sphere_radius = (dome_radius**2 + dome_height**2) / (2 * dome_height)

    half_width = int(dome_radius / dx)
    width = 2 * half_width + 1
    height_map = np.zeros((width, width))
    X, Y = np.meshgrid(np.linspace(-half_width * dx, half_width * dx, width),
                       np.linspace(-half_width * dx, half_width * dx, width))
    height_map = np.clip(
        np.sqrt(np.clip(sphere_radius**2 -
                        (X**2 + Y**2), a_min=0, a_max=None)) -
        (sphere_radius - dome_height),
        a_min=0,
        a_max=None)
    return height_map


@dataclass
class SampleActionOptions:
    position_std: float = 0.1
    height_std: float = 0.03


def sample_action(env: rolling_env.RollingEnv,
                  options: SampleActionOptions) -> np.ndarray:
    """
    Generate a sampled action for the RollingPin environment.
    """
    while True:
        if env.action_option == rolling_env.ActionOption.LIFTAFTERROLL:
            duration = 1.
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
            action = np.concatenate(([duration], start_position, [
                start_yaw, start_pitch, rolling_distance, delta_height,
                delta_yaw, delta_pitch
            ]))
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
        options: GenerateTransitionTupleOptions) -> TransitionTuple:
    """
    For the current dough rolling environemnt, sample an action, and then
    generate the tuple (current_state, action, next_state)
    """
    current_state = env.calc_heightmap(options.dx, options.grid_size)
    action = sample_action(env, options.sample_action_options)
    env.step(action)
    next_state = env.calc_heightmap(options.dx, options.grid_size)
    return TransitionTuple(current_state, action, next_state)


@dataclass
class TransitionTupleDatasetOptions:
    dtype = torch.float


class TransitionTupleDataset(torch.utils.data.TensorDataset):
    # I will assume all heightmaps are generated on the same grid.
    grid_h: torch.Tensor
    grid_w: torch.Tensor

    def __init__(self, transition_tuples: typing.List[TransitionTuple],
                 options: TransitionTupleDatasetOptions):
        self.grid_h = torch.from_numpy(
            transition_tuples[0].current_state.grid_h).to(options.dtype)
        self.grid_w = torch.from_numpy(
            transition_tuples[0].current_state.grid_w).to(options.dtype)

        current_heights = torch.empty(
            (len(transition_tuples),
             *transition_tuples[0].current_state.height.shape),
            dtype=options.dtype)
        actions = torch.empty(
            (len(transition_tuples), *transition_tuples[0].action.shape),
            dtype=options.dtype)
        next_heights = torch.empty(
            (len(transition_tuples),
             *transition_tuples[0].next_state.height.shape),
            dtype=options.dtype)
        for i, transition_tuple in enumerate(transition_tuples):
            # Check if the heightmaps are generated on the same grid.
            assert (np.array_equal(transition_tuple.current_state.grid_w,
                                   transition_tuples[0].current_state.grid_w))
            assert (np.array_equal(transition_tuple.current_state.grid_h,
                                   transition_tuples[0].current_state.grid_h))
            assert (np.array_equal(transition_tuple.next_state.grid_w,
                                   transition_tuples[0].current_state.grid_w))
            assert (np.array_equal(transition_tuple.next_state.grid_h,
                                   transition_tuples[0].current_state.grid_h))
            current_heights[i] = torch.from_numpy(
                transition_tuple.current_state.height).to(options.dtype)
            actions[i] = torch.from_numpy(transition_tuple.action).to(
                options.dtype)
            next_heights[i] = torch.from_numpy(
                transition_tuple.next_state.height).to(options.dtype)
        super(TransitionTupleDataset, self).__init__(current_heights, actions,
                                                     next_heights)
