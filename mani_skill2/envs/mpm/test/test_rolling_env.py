import unittest

import numpy as np
import sapien.core as sapien
import transforms3d

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

    def test_step(self):
        dut = rolling_env.RollingEnv(sim_freq=500, mpm_freq=2000)
        dut.agent.robot.set_pose(
            sapien.Pose(p=np.array([0, 0, 0.1]),
                        q=np.array([np.cos(0.1), 0, 0,
                                    np.sin(0.1)])))
        X_Wpin_init = dut.agent.robot.get_pose()
        duration = 1.
        rolling_distance = 0.2
        delta_height = 0.01
        delta_yaw = 0.
        delta_pitch = 0.
        action = np.array(
            [duration, rolling_distance, delta_height, delta_yaw, delta_pitch])
        dut.step(action)
        X_Wpin_final = dut.agent.robot.get_pose()
        np.testing.assert_allclose(X_Wpin_init.q, X_Wpin_final.q)

        # Check the rolling distance.
        self.assertAlmostEqual((X_Wpin_final.p - X_Wpin_init.p).dot(
            transforms3d.quaternions.quat2mat(X_Wpin_init.q)[:, 1]),
                               rolling_distance)
        # Check the height
        self.assertAlmostEqual(X_Wpin_final.p[2],
                               X_Wpin_init.p[2] + delta_height)

        # Now change the yaw and the pitch
        delta_yaw = 0.1
        delta_pitch = -0.2
        action = np.array(
            [duration, rolling_distance, delta_height, delta_yaw, delta_pitch])
        X_Wpin_init = dut.agent.robot.get_pose()
        dut.step(action)
        X_Wpin_final = dut.agent.robot.get_pose()
        arc_radius = rolling_distance / delta_yaw
        # Compute the position of the final pin pose expressed in the initial frame.
        p_InitFinal = transforms3d.quaternions.quat2mat(
            X_Wpin_init.q).T @ (X_Wpin_final.p - X_Wpin_init.p)
        np.testing.assert_allclose(p_InitFinal,
                                   np.array([
                                       arc_radius * (1 - np.cos(delta_yaw)),
                                       arc_radius * np.sin(delta_yaw),
                                       delta_height
                                   ]),
                                   atol=1E-5)
        np.testing.assert_allclose(
            transforms3d.quaternions.quat2mat(X_Wpin_final.q),
            transforms3d.quaternions.quat2mat(X_Wpin_init.q)
            @ transforms3d.euler.euler2mat(0, delta_pitch, delta_yaw),
            atol=1E-8)


if __name__ == "__main__":
    unittest.main()
