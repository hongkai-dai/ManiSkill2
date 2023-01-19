import examples.dough_model_learning.data_generation as data_generation

import unittest

import numpy as np
import torch

import mani_skill2.envs.mpm.rolling_env as rolling_env


class TestGenerateDomeHeightmap(unittest.TestCase):

    def test(self):
        dome_radius = 0.1
        dome_height = 0.05
        dx = 0.001
        height_map = data_generation.generate_dome_heightmap(
            dome_radius, dome_height, dx)
        self.assertAlmostEqual(height_map.max(), dome_height)
        self.assertEqual(height_map.min(), 0)

        # Test the height at a random position.
        x_index = 70
        y_index = 90
        half_width = int(dome_radius / dx)
        xy_coordinate = np.array([(half_width - x_index) * dx,
                                  (half_width - y_index) * dx])
        xy_height = height_map[x_index, y_index]

        sphere_radius = (dome_radius**2 + dome_height**2) / (2 * dome_height)
        self.assertAlmostEqual(
            (sphere_radius - dome_height)**2 + dome_radius**2,
            sphere_radius**2)
        self.assertAlmostEqual((xy_height + (sphere_radius - dome_height))**2 +
                               (xy_coordinate**2).sum(), sphere_radius**2)


class TestGenerateTransitionTuple(unittest.TestCase):

    def test(self):
        env = rolling_env.RollingEnv(sim_freq=500, mpm_freq=2000)
        height_map = data_generation.generate_dome_heightmap(dome_radius=0.1,
                                                             dome_height=0.05,
                                                             dx=0.005)
        env.set_initial_height_map(height_map, dx=0.005)

        options = data_generation.GenerateTransitionTupleOptions()
        transition_tuple = data_generation.generate_transition_tuple(
            env, options)


class TestTransitionTupleDataset(unittest.TestCase):

    def test(self):
        env = rolling_env.RollingEnv(sim_freq=500, mpm_freq=2000)
        height_map = data_generation.generate_dome_heightmap(dome_radius=0.1,
                                                             dome_height=0.05,
                                                             dx=0.005)
        env.set_initial_height_map(height_map, dx=0.005)

        generate_options = data_generation.GenerateTransitionTupleOptions()
        transition_tuples = []
        num_tuples = 5
        for _ in range(num_tuples):
            transition_tuples.append(
                data_generation.generate_transition_tuple(
                    env, generate_options))

        dataset = data_generation.TransitionTupleDataset(
            transition_tuples, dtype=torch.float64)
        self.assertEqual(len(dataset), num_tuples)

        # Get one item from the dataset, check the size of the data.
        current_height, action, next_height = dataset[0]
        self.assertEqual(current_height.shape,
                         (len(dataset.grid_h), len(dataset.grid_w)))
        self.assertEqual(next_height.shape,
                         (len(dataset.grid_h), len(dataset.grid_w)))


if __name__ == "__main__":
    unittest.main()
