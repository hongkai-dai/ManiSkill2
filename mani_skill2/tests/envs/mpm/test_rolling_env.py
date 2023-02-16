import functools
import unittest

import numpy as np
import pytest
import sapien.core as sapien
import transforms3d

import mani_skill2.envs.mpm.rolling_env as rolling_env
from mani_skill2.envs.mpm.rolling_env import generate_dome_heightmap


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestRollingEnv(unittest.TestCase):
    def test_step_action_one_way_coupling(self):
        dut = rolling_env.RollingEnv(sim_freq=100, mpm_freq=2000)

        pin_init_q = np.array([0, 0, 0.05 + dut.agent.capsule_radius, 0, 0, 0, 1])
        dut.agent.reset(pin_init_q)
        next_pose = sapien.Pose(
            p=pin_init_q[:3] - np.array([0, 0, 0.01]), q=np.array([1, 0, 0, 0])
        )

        dut.step_action_one_way_coupling(next_pose)
        np.testing.assert_almost_equal(
            dut.agent.robot.get_pose().p,
            pin_init_q[:3] - np.array([0, 0, 0.01]),
            decimal=7,
        )
        np.testing.assert_equal(dut.agent.robot.get_pose().q, np.array([1, 0, 0, 0.0]))

    def test_calc_heightmap(self):
        dut = rolling_env.RollingEnv(sim_freq=100, mpm_freq=2000)
        height = np.array(
            [
                [0.02, 0.04, 0.05, 0.03],
                [0.02, 0.01, 0.03, 0.04],
                [0.01, 0.03, 0.02, 0.05],
            ]
        )
        dx = 0.0025
        dut.set_initial_height_map(height, dx)
        dut.reset(regenerate_height_map=False)

        height_map = dut.calc_heightmap(dx, height.shape)
        # MPMBuilder.add_mpm_from_height_map actually only set the height to
        # `height` - dx, instead of the desired height.
        np.testing.assert_allclose(height - dx, height_map.height)
        self.assertEqual(height_map.grid_h.shape, (3,))
        self.assertEqual(height_map.grid_w.shape, (4,))
        np.testing.assert_allclose(
            height_map.grid_h[1:] - height_map.grid_h[:-1], dx * np.ones((2,))
        )
        np.testing.assert_allclose(
            height_map.grid_w[1:] - height_map.grid_w[:-1], dx * np.ones((3,))
        )

    def test_step_relative(self):
        dut = rolling_env.RollingEnv(sim_freq=500, mpm_freq=2000)
        dut.agent.robot.set_pose(
            sapien.Pose(
                p=np.array([0, 0, 0.1]), q=np.array([np.cos(0.1), 0, 0, np.sin(0.1)])
            )
        )
        X_Wpin_init = dut.agent.robot.get_pose()
        duration = 1.0
        rolling_distance = 0.2
        delta_height = 0.01
        delta_yaw = 0.0
        delta_pitch = 0.0
        action = np.array(
            [duration, rolling_distance, delta_height, delta_yaw, delta_pitch]
        )
        dut.step_relative(action)
        X_Wpin_final = dut.agent.robot.get_pose()
        np.testing.assert_allclose(X_Wpin_init.q, X_Wpin_final.q)

        # Check the rolling distance.
        self.assertAlmostEqual(
            (X_Wpin_final.p - X_Wpin_init.p).dot(
                transforms3d.quaternions.quat2mat(X_Wpin_init.q)[:, 1]
            ),
            rolling_distance,
        )
        # Check the height
        self.assertAlmostEqual(X_Wpin_final.p[2], X_Wpin_init.p[2] + delta_height)

        # Now change the yaw and the pitch
        delta_yaw = 0.1
        delta_pitch = -0.2
        action = np.array(
            [duration, rolling_distance, delta_height, delta_yaw, delta_pitch]
        )
        X_Wpin_init = dut.agent.robot.get_pose()
        dut.step_relative(action)
        X_Wpin_final = dut.agent.robot.get_pose()
        arc_radius = rolling_distance / delta_yaw
        # Compute the position of the final pin pose expressed in the initial frame.
        p_InitFinal = transforms3d.quaternions.quat2mat(X_Wpin_init.q).T @ (
            X_Wpin_final.p - X_Wpin_init.p
        )
        np.testing.assert_allclose(
            p_InitFinal,
            np.array(
                [
                    arc_radius * (1 - np.cos(delta_yaw)),
                    arc_radius * np.sin(delta_yaw),
                    delta_height,
                ]
            ),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            transforms3d.quaternions.quat2mat(X_Wpin_final.q),
            transforms3d.quaternions.quat2mat(X_Wpin_init.q)
            @ transforms3d.euler.euler2mat(0, delta_pitch, delta_yaw),
            atol=1e-8,
        )

    def test_step_absolute(self):
        dut = rolling_env.RollingEnv(sim_freq=500, mpm_freq=2000)
        dut.agent.robot.set_pose(
            sapien.Pose(
                p=np.array([0, 0, 0.1]), q=np.array([np.cos(0.1), 0, 0, np.sin(0.1)])
            )
        )

        duration = 2
        p_Wpin_command = np.array([0, 0.1, 0.07])
        q_Wpin_command = np.array([np.cos(-0.1), 0, 0, np.sin(-0.1)])
        action = np.concatenate(([duration], p_Wpin_command, q_Wpin_command))
        dut.step_absolute(action)

        pose_final = dut.agent.robot.get_pose()
        np.testing.assert_allclose(pose_final.p, p_Wpin_command, atol=1e-7)
        np.testing.assert_allclose(
            transforms3d.quaternions.quat2mat(pose_final.q),
            transforms3d.quaternions.quat2mat(q_Wpin_command),
        )

    def test_step_with_lift(self):
        dut = rolling_env.RollingEnv(sim_freq=500, mpm_freq=2000)
        dut.agent.robot.set_pose(
            sapien.Pose(
                p=np.array([0, 0, 0.1]), q=np.array([np.cos(0.1), 0, 0, np.sin(0.1)])
            )
        )

        duration = 2
        start_sweeping_pose = np.array([0, 0, 0.04, 0.01, 0.0])
        rolling_distance = 0.2
        delta_height = 0.01
        delta_yaw = 0.05
        delta_pitch = 0.1
        dut.step_with_lift(
            np.concatenate(
                (
                    [duration],
                    start_sweeping_pose,
                    [rolling_distance, delta_height, delta_yaw, delta_pitch],
                )
            )
        )

    def test_is_action_valid(self):
        dut = rolling_env.RollingEnv(sim_freq=500, mpm_freq=2000)
        dut.action_option = rolling_env.ActionOption.LIFTAFTERROLL
        self.assertTrue(
            dut.is_action_valid(
                np.array([0.1, 0, 0, 0.05, 0.2, 0.01, 0.1, 0.01, 0.1, 0.01])
            )
        )
        # duration is negative.
        self.assertFalse(
            dut.is_action_valid(np.array([-0.1, 0, 0, 0.05, 0, 0, 0.1, 0, 0, 0]))
        )
        # The pin is below the table at the start pose.
        self.assertFalse(
            dut.is_action_valid(np.array([0.5, 0, 0, 0.01, 0, 0.2, 0.5, 0.4, 0, -0.2]))
        )
        # The pin is below the table at the end pose.
        self.assertFalse(
            dut.is_action_valid(np.array([0.5, 0, 0, 0.05, 0, 0, 0.1, -0.03, 0, 0.2]))
        )

    def test_dough_center(self):
        dut = rolling_env.RollingEnv(sim_freq=500, mpm_freq=2000)
        height = 0.5
        dx = 0.1
        dut.set_initial_height_map(
            np.array([[0, 0, 0], [0, height, height], [0, height, height]]), dx
        )
        dut.reset(regenerate_height_map=False)
        # Currently when we initialize the height map, the top cell
        # (with size dx * dx * dx) is always not filled. I should fix this bug.
        np.testing.assert_allclose(
            dut.dough_center(), np.array([dx / 2, dx / 2, (height - dx) / 2])
        )

    def test_step(self):
        dough_initializer = functools.partial(
            rolling_env.generate_circular_cone_heightmap,
            radius=0.1,
            height=0.06,
        )
        dut = rolling_env.RollingEnv(
            sim_freq=500,
            mpm_freq=2000,
            dough_initializer=dough_initializer,
        )
        dut.reset()

        duration = 2
        start_sweeping_pose = np.array([0, 0, 0.04, 0.01, 0.0])
        rolling_distance = 0.2
        delta_height = 0.01
        delta_yaw = 0.05
        delta_pitch = 0.1
        action = np.concatenate(
            (
                [duration],
                start_sweeping_pose,
                [rolling_distance, delta_height, delta_yaw, delta_pitch],
            )
        )

        obs, rew, done, info = dut.step(action)


class TestGenerateDomeHeightmap(unittest.TestCase):
    def test(self):
        dome_radius = 0.1
        dome_height = 0.05
        dx = 0.001
        height_map = generate_dome_heightmap(
            dome_radius,
            dome_height,
            dx,
        )
        self.assertAlmostEqual(height_map.max(), dome_height)
        self.assertEqual(height_map.min(), 0)

        # Test the height at a random position.
        x_index = 70
        y_index = 90
        half_width = int(dome_radius / dx)
        xy_coordinate = np.array(
            [(half_width - x_index) * dx, (half_width - y_index) * dx]
        )
        xy_height = height_map[x_index, y_index]

        sphere_radius = (dome_radius**2 + dome_height**2) / (2 * dome_height)
        self.assertAlmostEqual(
            (sphere_radius - dome_height) ** 2 + dome_radius**2, sphere_radius**2
        )
        self.assertAlmostEqual(
            (xy_height + (sphere_radius - dome_height)) ** 2
            + (xy_coordinate**2).sum(),
            sphere_radius**2,
        )


if __name__ == "__main__":
    unittest.main()
