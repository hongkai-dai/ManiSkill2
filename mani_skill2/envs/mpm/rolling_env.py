import typing
from dataclasses import dataclass

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat
import transforms3d
import warp as wp

from mani_skill2.agents.configs.panda.variants import PandaPinchConfig
from mani_skill2.agents.robots.rolling_pin import RollingPin
from mani_skill2.envs.mpm.base_env import MPMBaseEnv, MPMModelBuilder, MPMSimulator
from mani_skill2.utils.registration import register_gym_env


@dataclass
class Heightmap:
    """
    Records the grid coordinate in the horizontal plane and the height at each
    grid vertex.

    height[i, j] is the height at the horizontal coordinate (grid_w[i], grid_h[j])
    """
    grid_h: np.ndarray
    grid_w: np.ndarray
    height: np.ndarray


@register_gym_env("Rolling-v0", max_episode_steps=10000)
class RollingEnv(MPMBaseEnv):
    agent: RollingPin

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        self._height_map_dx = 0.0025
        self._initial_height_map = np.ones((10, 10)) * 0.01
        super().__init__(*args, **kwargs)

    def reset(self, *args, seed=None, **kwargs):
        return super().reset(*args, seed=seed, **kwargs)

    def _setup_mpm(self):
        self.model_builder = MPMModelBuilder()
        self.model_builder.set_mpm_domain(domain_size=[0.5, 0.5, 0.5],
                                          grid_length=0.02)
        self.model_builder.reserve_mpm_particles(count=self.max_particles)

        self._setup_mpm_bodies()

        self.mpm_simulator = MPMSimulator(device="cuda")
        self.mpm_model = self.model_builder.finalize(device="cuda")
        self.mpm_model.gravity = np.array((0.0, 0.0, -9.81), dtype=np.float32)
        self.mpm_model.struct.ground_normal = wp.vec3(0.0, 0.0, 1.0)
        self.mpm_model.struct.particle_radius = 0.005
        self.mpm_states = [
            self.mpm_model.state()
            for _ in range(self._mpm_step_per_sapien_step + 1)
        ]

    def set_initial_height_map(self, initial_height_map: np.ndarray,
                               dx: float):
        self._initial_height_map = initial_height_map
        self._height_map_dx = dx

    def _initialize_mpm(self):
        self.model_builder.clear_particles()

        # These constants are copied from a file I obtained from
        # ManiSkill2 authors.
        # I might need to tune the Young's modulus later.
        E = 2E5
        nu = 0.1
        mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
        ys = 1e3
        # 0 for von-mises. In Pinch-v0 it also uses type=0
        type = 0
        friction_angle = 0.5
        cohesion = 0.05
        height_map = self._initial_height_map
        height_map_dx = self._height_map_dx
        count = self.model_builder.add_mpm_from_height_map(
            pos=(0., 0., 0.),
            vel=(0., 0., 0.),
            dx=height_map_dx,
            height_map=height_map,
            density=1.4E3,
            mu_lambda_ys=(mu, lam, ys),
            friction_cohesion=(friction_angle, cohesion, 0.),
            type=type,
            jitter=False,
            color=(1., 0., 0.))

        self.model_builder.init_model_state(self.mpm_model, self.mpm_states)

        # I copied these values from pinch_env.py
        self.mpm_model.struct.static_ke = 100.0
        self.mpm_model.struct.static_kd = 0.5
        self.mpm_model.struct.static_mu = 1.0
        self.mpm_model.struct.static_ka = 0.0

        self.mpm_model.struct.body_ke = 1000.0
        self.mpm_model.struct.body_kd = 0.2
        self.mpm_model.struct.body_mu = 0.5
        self.mpm_model.struct.body_ka = 0.0

        self.mpm_model.adaptive_grid = True

        self.mpm_model.grid_contact = False
        self.mpm_model.particle_contact = True
        self.mpm_model.struct.body_sticky = False
        self.mpm_model.struct.ground_sticky = 1

        self.mpm_model.struct.particle_radius = 0.0025

    def _get_coupling_actors(self):
        # The robot (rolling pin) is the only actor.
        return [self.agent.robot]

    def _load_agent(self):
        self.agent = RollingPin(
            self._scene,
            self.control_freq,
        )

    def _initialize_agent(self):
        # The rolling pin capsule axis is along x-axis.
        # The position vector is
        # [p_x, p_y, p_z, quat_x, quat_y, quat_z, quat_w]
        qpos = np.array([0, 0., 0.1, 0, 0, 0, 1])

        self.agent.reset(qpos)

    def step(self, *args, **kwargs):
        return super().step(*args, **kwargs)

    def _setup_cameras(self):
        # Camera only for rendering, not included in `_cameras`
        self.render_camera = self._scene.add_camera("render_camera", 512, 512,
                                                    1, 0.001, 10)
        self.render_camera.set_local_pose(
            sapien.Pose([-0.05, 0.7, 0.3], euler2quat(0, np.pi / 10,
                                                      -np.pi / 2)))

        base_camera = self._scene.add_camera("base_camera", 128, 128,
                                             np.pi / 2, 0.01, 10)
        base_camera.set_local_pose(
            sapien.Pose([0.4, 0, 0.3], euler2quat(0, np.pi / 10, -np.pi)))

        self._cameras["base_camera"] = base_camera

    def step_action(self, scene_time: float, action: np.ndarray):
        """
        I overload the parent step_action method. In the parent method it
        assumes that the agent is a position-controller articulated robot.
        On the other hand, we will apply a wrench on the single rigid-body
        rolling pin directly.
        """
        self.agent.controller.set_goal(action)

        # The code below are adpated from MPMBaseEnv.step_action().
        for _ in range(self._sim_steps_per_control):
            self.sync_actors()
            for mpm_step in range(self._mpm_step_per_sapien_step):
                self.mpm_simulator.simulate(
                    self.mpm_model,
                    self.mpm_states[mpm_step],
                    self.mpm_states[mpm_step + 1],
                    self._mpm_dt,
                )
                scene_time += self._mpm_dt

            self.agent.before_simulation_step()

            # apply wrench
            tfs = [s.ext_body_f.numpy().copy() for s in self.mpm_states[:-1]]
            tfs = np.mean(tfs, 0)

            if np.isnan(tfs).any():
                self.sim_crashed = True
                return

            if self.mpm_states[-1].struct.error.numpy()[0] == 1:
                self.sim_crashed = True
                return

            # Compute the wrench from applied by the controller.
            controller_wrench = self.agent.controller.compute_wrench(
                scene_time, self.agent.robot.get_pose(),
                self.agent.robot.get_velocity(),
                self.agent.robot.get_angular_velocity())
            for actor, tf in zip(self._coupled_actors, tfs):
                if actor.type not in ["kinematic", "static"]:
                    if actor is self.agent.robot:
                        tf[3:] += controller_wrench[:3]
                        tf[:3] += controller_wrench[3:6]
                    actor.add_force_torque(tf[3:], tf[:3])

            self._scene.step()
            self.mpm_states = [self.mpm_states[-1]
                               ] + self.mpm_states[:-1]  # rotate states

    def step_action_one_way_coupling(self, next_pin_pose: sapien.Pose):
        """
        Only consider the one-way coupling, namely the rolling pin as a
        boundary condition for the dough, but ignore the reaction force applied
        from the dough to the rolling pin. Instead we assume that the rolling
        pin just follows a linearly interpolated trajectory from the current
        pose to `pin_next_pose`.
        """
        start_pose = self.agent.robot.get_pose()
        # Compute the delta between start_pose and next_pin_pose
        delta_position = (next_pin_pose.p -
                          start_pose.p) / self._sim_steps_per_control
        R_WB2 = transforms3d.quaternions.quat2mat(next_pin_pose.q)
        R_WB1 = transforms3d.quaternions.quat2mat(start_pose.q)
        axis, angle = transforms3d.axangles.mat2axangle(R_WB1.T @ R_WB2)
        delta_angle = angle / self._sim_steps_per_control

        for sim_step in range(self._sim_steps_per_control):

            self.sync_actors()
            for mpm_step in range(self._mpm_step_per_sapien_step):
                self.mpm_simulator.simulate(
                    self.mpm_model,
                    self.mpm_states[mpm_step],
                    self.mpm_states[mpm_step + 1],
                    self._mpm_dt,
                )

            self.agent.before_simulation_step()

            # apply wrench
            tfs = [s.ext_body_f.numpy() for s in self.mpm_states[:-1]]
            tfs = np.mean(tfs, 0)

            if np.isnan(tfs).any():
                self.sim_crashed = True
                return

            # Compute the wrench from applied by the controller.
            if self.mpm_states[-1].struct.error.numpy()[0] == 1:
                self.sim_crashed = True
                return

            self._scene.step()
            R_sim = R_WB1 @ transforms3d.axangles.axangle2mat(
                axis, delta_angle * (sim_step + 1))
            self.agent.robot.set_pose(
                sapien.Pose(p=start_pose.p + delta_position * (sim_step + 1),
                            q=transforms3d.quaternions.mat2quat(R_sim)))
            self.agent.robot.set_velocity(np.zeros(3))
            self.agent.robot.set_angular_velocity(np.zeros(3))
            self.mpm_states = [self.mpm_states[-1]
                               ] + self.mpm_states[:-1]  # rotate states

    def calc_heightmap(self, dx: float,
                       grid_size: typing.Tuple[int]) -> Heightmap:
        """
        Compute the heightmap of the dough in the planar box region.
        This box region is the grid with square grid cell (with length dx)
        -(grid_size[1] - 1) * dx / 2 <= p_x <= grid_size[1] * dx / 2
        -(grid_size[0] - 1) * dx / 2 <= p_y <= grid_size[0] * dx / 2
        The height map is reported on the grid with the given grid_size

        Args:
          dx: The length of each cell in the grid.
          grid_size: A size-2 list containing the (grid_size_h, grid_size_w)
        """
        grid_size_h = grid_size[0]
        grid_size_w = grid_size[1]

        grid_h = (np.arange(0, grid_size_h) - (grid_size_h - 1) / 2) * dx
        grid_w = (np.arange(0, grid_size_w) - (grid_size_w - 1) / 2) * dx
        height = np.zeros(grid_size)

        particle_q = self.copy_array_to_numpy(
            self.mpm_states[0].struct.particle_q,
            self.mpm_model.struct.n_particles)
        for i in range(particle_q.shape[0]):
            h_index = round(particle_q[i, 1] / dx + (grid_size_h - 1) / 2)
            w_index = round(particle_q[i, 0] / dx + (grid_size_w - 1) / 2)
            if h_index < 0:
                h_index = 0
            if h_index > grid_size_h - 1:
                h_index = grid_size_h
            if w_index < 0:
                w_index = 0
            if w_index > grid_size_w - 1:
                w_index = grid_size_w - 1
            if particle_q[i, 2] > height[h_index, w_index]:
                height[h_index, w_index] = particle_q[i, 2]
        return Heightmap(grid_h=grid_h, grid_w=grid_w, height=height)
