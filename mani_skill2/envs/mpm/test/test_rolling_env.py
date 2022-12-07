import unittest

import numpy as np
import sapien.core as sapien

import mani_skill2.envs.mpm.rolling_env as rolling_env


class TestRollingEnv(unittest.TestCase):

    def test_step_action_one_way_coupling(self):
        dut = rolling_env.RollingEnv(sim_freq=100, mpm_freq=2000)

        pin_init_q = np.array(
            [0, 0, 0.05 + dut.agent.capsule_radius, 0, 0, 0, 1])
        dut.agent.reset(pin_init_q)
        next_pose = sapien.Pose(p=pin_init_q[:3] - np.array([0, 0, 0.01]),
                                q=np.array([1, 0, 0, 0]))

        dut.step_action_one_way_coupling(next_pose)
        np.testing.assert_almost_equal(dut.agent.robot.get_pose().p,
                                       pin_init_q[:3] - np.array([0, 0, 0.01]),
                                       decimal=7)
        np.testing.assert_equal(dut.agent.robot.get_pose().q,
                                np.array([1, 0, 0, 0.]))

    def test_calc_heightmap(self):
        dut = rolling_env.RollingEnv(sim_freq=100, mpm_freq=2000)
        height = np.array([[0.02, 0.04, 0.05, 0.03], [0.02, 0.01, 0.03, 0.04],
                           [0.01, 0.03, 0.02, 0.05]])
        dx = 0.0025
        dut.set_initial_height_map(height, dx)
        dut.reset()

        height_map = dut.calc_heightmap(dx, height.shape)
        # MPMBuilder.add_mpm_from_height_map actually only set the height to `height` - dx, instead of the desired height.
        np.testing.assert_allclose(height - dx, height_map.height)
        self.assertEqual(height_map.grid_h.shape, (3, ))
        self.assertEqual(height_map.grid_w.shape, (4, ))
        np.testing.assert_allclose(
            height_map.grid_h[1:] - height_map.grid_h[:-1], dx * np.ones(
                (2, )))
        np.testing.assert_allclose(
            height_map.grid_w[1:] - height_map.grid_w[:-1], dx * np.ones(
                (3, )))


if __name__ == "__main__":
    unittest.main()
