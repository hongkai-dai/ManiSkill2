import examples.dough_model_learning.data_generation as data_generation

import unittest

import numpy as np

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


class TestGenerateStateActionPair(unittest.TestCase):

    def test(self):
        env = rolling_env.RollingEnv(sim_freq=500, mpm_freq=2000)
        height_map = data_generation.generate_dome_heightmap(dome_radius=0.1,
                                                             dome_height=0.05,
                                                             dx=0.005)
        env.set_initial_height_map(height_map, dx=0.005)

        options = data_generation.GenerateStateActionOptions()
        state_action_pair = data_generation.generate_state_action_pair(
            env, options)


if __name__ == "__main__":
    unittest.main()
