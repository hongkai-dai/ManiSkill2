import examples.dough_model_learning.data_generation as data_generation

import unittest

import numpy as np
import torch

import mani_skill2.envs.mpm.rolling_env as rolling_env
from mani_skill2.envs.mpm.rolling_env import generate_dome_heightmap


class TestGenerateTransitionTuple(unittest.TestCase):
    def test(self):
        env = rolling_env.RollingEnv(sim_freq=500, mpm_freq=2000)
        height_map = generate_dome_heightmap(
            dome_radius=0.1,
            dome_height=0.05,
            dx=0.005,
        )
        env.set_initial_height_map(height_map, dx=0.005)

        options = data_generation.GenerateTransitionTupleOptions()
        transition_tuple = data_generation.generate_transition_tuple(env, options)


class TestTransitionTupleDataset(unittest.TestCase):
    def test(self):
        env = rolling_env.RollingEnv(sim_freq=500, mpm_freq=2000)
        height_map = generate_dome_heightmap(
            dome_radius=0.1,
            dome_height=0.05,
            dx=0.005,
        )
        env.set_initial_height_map(height_map, dx=0.005)

        generate_options = data_generation.GenerateTransitionTupleOptions()
        transition_tuples = []
        num_tuples = 5
        for _ in range(num_tuples):
            transition_tuples.append(
                data_generation.generate_transition_tuple(env, generate_options)
            )

        dataset = data_generation.TransitionTupleDataset(
            transition_tuples, dtype=torch.float64
        )
        self.assertEqual(len(dataset), num_tuples)

        # Get one item from the dataset, check the size of the data.
        current_height, action, next_height = dataset[0]
        self.assertEqual(
            current_height.shape, (len(dataset.grid_h), len(dataset.grid_w))
        )
        self.assertEqual(next_height.shape, (len(dataset.grid_h), len(dataset.grid_w)))


if __name__ == "__main__":
    unittest.main()
