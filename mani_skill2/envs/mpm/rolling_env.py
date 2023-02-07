from dataclasses import dataclass
from enum import Enum
import typing
from typing import Callable, Tuple

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat
import transforms3d
import warp as wp

from mani_skill2.agents.configs.panda.variants import PandaPinchConfig
from mani_skill2.agents.robots.rolling_pin import RollingPin
from mani_skill2.envs.mpm.base_env import MPMBaseEnv, MPMModelBuilder, MPMSimulator
from mani_skill2.utils.registration import register_gym_env


class ActionOption(Enum):
    RELATIVE = 1
    ABSOLUTE = 2
    LIFTAFTERROLL = 3  # Lift the rolling pin after every rolling motion.


@dataclass
class Heightmap:
    """Records the grid coordinate in the horizontal plane and the height at each grid vertex.

    height[i, j] is the height at the horizontal coordinate (grid_w[i], grid_h[j])
    """

    grid_h: np.ndarray
    grid_w: np.ndarray
    height: np.ndarray


def generate_circular_cone_heightmap(
    radius: float,
    height: float,
    dx: float,
) -> np.ndarray:
    half_width = int(radius / dx)
    width = 2 * half_width + 1
    height_map = np.zeros((width, width), dtype=np.float32)
    X, Y = np.meshgrid(
        np.linspace(-half_width * dx, half_width * dx, width),
        np.linspace(-half_width * dx, half_width * dx, width),
    )
    height_map = height - np.sqrt(X ** 2 + Y ** 2) / radius * height
    height_map = np.clip(height_map, a_min=np.zeros_like(height_map), a_max=None)
    return height_map


def generate_dome_heightmap(
    dome_radius: float,
    dome_height: float,
    dx: float,
) -> np.ndarray:
    """Generate the initial heightmap as a dome.

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
    sphere_radius = (dome_radius ** 2 + dome_height ** 2) / (2 * dome_height)

    half_width = int(dome_radius / dx)
    width = 2 * half_width + 1
    height_map = np.zeros((width, width))
    X, Y = np.meshgrid(
        np.linspace(-half_width * dx, half_width * dx, width),
        np.linspace(-half_width * dx, half_width * dx, width),
    )
    height_map = np.clip(
        np.sqrt(np.clip(sphere_radius ** 2 - (X ** 2 + Y ** 2), a_min=0, a_max=None))
        - (sphere_radius - dome_height),
        a_min=0,
        a_max=None,
    )
    return height_map


@register_gym_env("Rolling-v0", max_episode_steps=10000)
class RollingEnv(MPMBaseEnv):
    agent: RollingPin
    action_option: ActionOption
    dough_initializer: Callable
    height_map_dx: float
    obs_height_map_dx: float
    obs_height_map_grid_size: Tuple[int, int]

    def __init__(
        self,
        *args,
        action_option=ActionOption.LIFTAFTERROLL,
        dough_initializer=None,
        height_map_dx=0.0025,
        obs_height_map_dx=0.01,
        obs_height_map_grid_size=(32, 32),
        **kwargs,
    ) -> None:
        """
        Args:
            action_option: The mode for applying the action.
            dough_initializer: Callable that returns a heightmap given dx.
            height_map_dx: The dx to use for the height map internally.
            obs_height_map_dx: The dx to use for observation dx.
            obs_height_map_grid_size: The grid size to use for the observation.
        """
        self.action_option = action_option
        self._dough_initializer = dough_initializer
        self._height_map_dx = height_map_dx
        self._obs_height_map_dx = obs_height_map_dx
        self._obs_height_map_grid_size = obs_height_map_grid_size

        # This is set to a non-None value b/c it used during the base class init.
        self._initial_height_map = np.ones((10, 10)) * 0.01

        super().__init__(*args, **kwargs)

    def reset(self, *args, seed=None, **kwargs):
        if self._dough_initializer is not None:
            self._initial_height_map = self._dough_initializer(dx=self._height_map_dx)
        return super().reset(*args, seed=seed, **kwargs)

    def _setup_mpm(self):
        self.model_builder = MPMModelBuilder()
        self.model_builder.set_mpm_domain(domain_size=[0.5, 0.5, 0.5], grid_length=0.02)
        self.model_builder.reserve_mpm_particles(count=self.max_particles)

        self._setup_mpm_bodies()

        self.mpm_simulator = MPMSimulator(device="cuda")
        self.mpm_model = self.model_builder.finalize(device="cuda")
        self.mpm_model.gravity = np.array((0.0, 0.0, -9.81), dtype=np.float32)
        self.mpm_model.struct.ground_normal = wp.vec3(0.0, 0.0, 1.0)
        self.mpm_model.struct.particle_radius = 0.005
        self.mpm_states = [
            self.mpm_model.state() for _ in range(self._mpm_step_per_sapien_step + 1)
        ]

    def set_initial_height_map(self, initial_height_map: np.ndarray, dx: float):
        self._initial_height_map = initial_height_map
        self._height_map_dx = dx

    def _initialize_mpm(self):
        self.model_builder.clear_particles()

        # These constants are copied from a file I obtained from
        # ManiSkill2 authors.
        # I might need to tune the Young's modulus later.
        E = 2e5
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
            pos=(0.0, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            dx=height_map_dx,
            height_map=height_map,
            density=1.4e3,
            mu_lambda_ys=(mu, lam, ys),
            friction_cohesion=(friction_angle, cohesion, 0.0),
            type=type,
            jitter=False,
            color=(1.0, 0.0, 0.0),
        )

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
        qpos = np.array([0, 0.0, 0.1, 0, 0, 0, 1])

        self.agent.reset(qpos)

    def _setup_cameras(self):
        # Camera only for rendering, not included in `_cameras`
        self.render_camera = self._scene.add_camera(
            "render_camera", 512, 512, 1, 0.001, 10
        )
        self.render_camera.set_local_pose(
            sapien.Pose([-0.05, 0.7, 0.3], euler2quat(0, np.pi / 10, -np.pi / 2))
        )

        base_camera = self._scene.add_camera(
            "base_camera", 128, 128, np.pi / 2, 0.01, 10
        )
        base_camera.set_local_pose(
            sapien.Pose([0.4, 0, 0.3], euler2quat(0, np.pi / 10, -np.pi))
        )

        self._cameras["base_camera"] = base_camera

    def step_action(self, scene_time: float, action: np.ndarray):
        """Overload of the parent step_action method.

        In the parent method it assumes that the agent
        is a position-controller articulated robot.
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
                scene_time,
                self.agent.robot.get_pose(),
                self.agent.robot.get_velocity(),
                self.agent.robot.get_angular_velocity(),
            )
            for actor, tf in zip(self._coupled_actors, tfs):
                if actor.type not in ["kinematic", "static"]:
                    if actor is self.agent.robot:
                        tf[3:] += controller_wrench[:3]
                        tf[:3] += controller_wrench[3:6]
                    actor.add_force_torque(tf[3:], tf[:3])

            self._scene.step()
            self.mpm_states = [self.mpm_states[-1]] + self.mpm_states[
                :-1
            ]  # rotate states

    def step_action_one_way_coupling(self, next_pin_pose: sapien.Pose):
        """Only consider the one-way coupling.

        Namely the rolling pin as a boundary condition for the dough,
        but ignore the reaction force applied from the dough to the rolling pin.
        Instead we assume that the rolling pin just follows a linearly
        interpolated trajectory from the current pose to `pin_next_pose`.
        """
        start_pose = self.agent.robot.get_pose()
        # Compute the delta between start_pose and next_pin_pose
        delta_position = (next_pin_pose.p - start_pose.p) / self._sim_steps_per_control
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
                axis, delta_angle * (sim_step + 1)
            )
            self.agent.robot.set_pose(
                sapien.Pose(
                    p=start_pose.p + delta_position * (sim_step + 1),
                    q=transforms3d.quaternions.mat2quat(R_sim),
                )
            )
            self.agent.robot.set_velocity(np.zeros(3))
            self.agent.robot.set_angular_velocity(np.zeros(3))
            self.mpm_states = [self.mpm_states[-1]] + self.mpm_states[
                :-1
            ]  # rotate states

    def calc_heightmap(self, dx: float, grid_size: typing.Tuple[int]) -> Heightmap:
        """Compute the heightmap of the dough in the planar box region.

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
            self.mpm_states[0].struct.particle_q, self.mpm_model.struct.n_particles
        )
        for i in range(particle_q.shape[0]):
            h_index = round(particle_q[i, 1] / dx + (grid_size_h - 1) / 2)
            w_index = round(particle_q[i, 0] / dx + (grid_size_w - 1) / 2)
            if h_index < 0:
                h_index = 0
            if h_index > grid_size_h - 1:
                h_index = grid_size_h - 1
            if w_index < 0:
                w_index = 0
            if w_index > grid_size_w - 1:
                w_index = grid_size_w - 1
            if particle_q[i, 2] > height[h_index, w_index]:
                height[h_index, w_index] = particle_q[i, 2]
        return Heightmap(grid_h=grid_h, grid_w=grid_w, height=height)

    def is_pin_above_table(self, q: typing.Union[np.ndarray, sapien.Pose]) -> bool:
        if isinstance(q, np.ndarray):
            p_Wpin = q[:3]
            quat = np.array([q[6], q[0], q[1], q[2]])
        elif isinstance(q, sapien.Pose):
            p_Wpin = q.p
            quat = q.q
        R_Wpin = transforms3d.quaternions.quat2mat(quat)
        if (p_Wpin + R_Wpin @ np.array([self.agent.capsule_half_length, 0, 0]))[
            2
        ] < self.agent.capsule_radius:
            return False
        if (p_Wpin + R_Wpin @ np.array([-self.agent.capsule_half_length, 0, 0]))[
            2
        ] < self.agent.capsule_radius:
            return False
        return True

    def is_pin_y_horizontal(self, R_Wpin: np.ndarray, tol: float = 0.0) -> bool:
        """Return true if the pin y axis is parallel to the horizontal surface

        (namely the pin y axis is orthogonal to world z axis)
        """
        return np.abs(R_Wpin[2, 1]) <= tol

    def _step_to_pose(
        self,
        duration: float,
        p_Wpin_final: np.ndarray,
        R_Wpin_final: np.ndarray,
    ) -> None:
        """Step to a commanded rolling pin pose from the current state within a given duration."""
        # Compute the number of control steps
        num_control_steps = int(duration * self._control_freq)

        pose_init = self.agent.robot.get_pose()
        p_Wpin_init = pose_init.p
        R_Wpin_init = transforms3d.quaternions.quat2mat(pose_init.q)

        # Now linearly interpolate between pin start pose and final pose, and
        # step the rolling pin controller to follow this interpolated
        # trajectory.
        delta_pos_step = (p_Wpin_final - p_Wpin_init) / num_control_steps
        axis, angle = transforms3d.axangles.mat2axangle(R_Wpin_init.T @ R_Wpin_final)
        delta_angle_step = angle / num_control_steps
        R_step = transforms3d.axangles.axangle2mat(axis, delta_angle_step)
        quat_step = transforms3d.quaternions.mat2quat(R_step)

        p_Wpin_next = p_Wpin_init.copy()
        quat_Wpin_next = transforms3d.quaternions.mat2quat(R_Wpin_init)
        for _ in range(num_control_steps):
            p_Wpin_next += delta_pos_step
            quat_Wpin_next = transforms3d.quaternions.qmult(quat_Wpin_next, quat_step)
            self.step_action_one_way_coupling(
                sapien.Pose(p=p_Wpin_next, q=quat_Wpin_next)
            )

    def _compute_step_relative_pose(
        self,
        start_pose: sapien.Pose,
        rolling_distance: float,
        delta_height: float,
        delta_yaw: float,
        delta_pitch: float,
    ) -> sapien.Pose:
        """
        Compute the final pose if we sweep the rolling pin from start_pose through certain motion.
        """
        p_Wpin_init = start_pose.p
        R_Wpin_init = transforms3d.quaternions.quat2mat(start_pose.q)
        if not self.is_pin_y_horizontal(R_Wpin_init, 1e-3):
            raise Exception(
                "RollingPinEnv.step(): pin y axis should be perpendicular to world z axis."
            )

        # Now compute the final pose.
        # First acount for rolling for the given distance with the turning angle.
        # We assume the path of the rolling pin center, projected onto the
        # table horizontal plane, is an arc, with arc radius determined by the
        # delta yaw angle.
        if delta_yaw == 0.0:
            # No turning, move a straight line along pin's y axis.
            p_Wpin_final = p_Wpin_init + R_Wpin_init @ np.array(
                [0, rolling_distance, 0]
            )
        else:
            # The angle corresponding with this arc is delta_yaw, with arc length rolling_distance
            arc_radius = rolling_distance / delta_yaw
            p_Wpin_final = p_Wpin_init + R_Wpin_init @ np.array(
                [
                    arc_radius * (1 - np.cos(delta_yaw)),
                    arc_radius * np.sin(delta_yaw),
                    0,
                ]
            )
        # Now account for delta_height
        p_Wpin_final[2] += delta_height

        # Now account for orientation change.
        R_Wpin_final = R_Wpin_init @ transforms3d.euler.euler2mat(
            0, delta_pitch, delta_yaw
        )
        return sapien.Pose(
            p=p_Wpin_final, q=transforms3d.quaternions.mat2quat(R_Wpin_final)
        )

    def step_relative(self, action: np.ndarray) -> None:
        """Perform a relative action step.

        The action is
        (duration, rolling_distance, delta_height, delta_yaw, delta_pitch)
        where duration is a positive scalar.
        rolling_distance is a scalar measuring the distance travelled along the
        rolling direction by the rolling pin center.
        delta_height is the change on the z height of the rolling pin.
        delta_yaw is the change of the yaw angle of the rolling pin.
        delta_pitch is the change of the pitch angle of the rolling pin (the
        rolling pin capsule axis is its x axis.)

        The robot will follow a linearly-interpolated trajectory from the
        starting pose to the final pose.
        """
        if action.shape != (5,):
            raise Exception(
                f"step_relative() expects action.shape==(5,), got {action.shape}"
            )
        duration, rolling_distance, delta_height, delta_yaw, delta_pitch = action

        X_Wpin_init = self.agent.robot.get_pose()
        X_Wpin_final = self._compute_step_relative_pose(
            X_Wpin_init, rolling_distance, delta_height, delta_yaw, delta_pitch
        )

        if not self.is_pin_above_table(X_Wpin_final):
            raise Exception(
                "RollingPinEnv.step(): the rolling pin is not above the table in the commanded final pose."
            )

        self._step_to_pose(
            duration, X_Wpin_final.p, transforms3d.quaternions.quat2mat(X_Wpin_final.q)
        )

    def step_absolute(self, action: np.ndarray) -> None:
        """Step from the current pose to a commanded pose.

        The action is [duration, command_pin_position, command_pin_quaternion]
        """
        self._step_to_pose(
            duration=action[0],
            p_Wpin_final=action[1:4],
            R_Wpin_final=transforms3d.quaternions.quat2mat(action[4:8]),
        )

    def step_with_lift(self, action: np.ndarray) -> None:
        """Descend the rolling pin to a desired pose, sweep the rolling pin, and then lift the rolling pin.

        The action is
        [duration, start_sweeping_pose, rolling distance, delta_height, delta_yaw, delta_pitch]
        start_sweeing_pose is [pos_x, pos_y, pos_z, yaw, pitch]
        """
        if action.shape != (10,):
            raise Exception(
                f"step_with_lift() expects action.shape=(10,), got {action.shape}"
            )

        (
            duration,
            start_sweeping_pose,
            rolling_distance,
            delta_height,
            delta_yaw,
            delta_pitch,
        ) = np.split(action, [1, 6, 7, 8, 9])
        start_sweeping_yaw = start_sweeping_pose[3]
        start_sweeping_pitch = start_sweeping_pose[4]

        # First move to a position right above start_sweeping_pose
        descend_height = 0.05
        R_Wpin_start_sweeping = transforms3d.euler.euler2mat(
            0, start_sweeping_pitch, start_sweeping_yaw
        )
        self._step_to_pose(
            duration=1,
            p_Wpin_final=np.array(
                [
                    start_sweeping_pose[0],
                    start_sweeping_pose[1],
                    start_sweeping_pose[2] + descend_height,
                ]
            ),
            R_Wpin_final=R_Wpin_start_sweeping,
        )
        # Descend to start_sweeping_pose.
        self._step_to_pose(
            duration=0.5,
            p_Wpin_final=start_sweeping_pose[:3],
            R_Wpin_final=R_Wpin_start_sweeping,
        )

        # Now sweep the rolling pin
        self.step_relative(
            np.concatenate(
                (duration, rolling_distance, delta_height, delta_yaw, delta_pitch)
            )
        )

        # Now lift up the rolling pin from the current pose.
        X_Wpin = self.agent.robot.get_pose()
        self._step_to_pose(
            duration=0.5,
            p_Wpin_final=X_Wpin.p + np.array([0, 0, descend_height]),
            R_Wpin_final=transforms3d.quaternions.quat2mat(X_Wpin.q),
        )

    def step(self, action: np.ndarray) -> None:
        if self.action_option == ActionOption.RELATIVE:
            self.step_relative(action)
        elif self.action_option == ActionOption.ABSOLUTE:
            self.step_absolute(action)
        elif self.action_option == ActionOption.LIFTAFTERROLL:
            self.step_with_lift(action)

        # TODO(blake.wulfe): Fill these in.
        rew = 0
        done = False
        info = {}
        return self._get_obs(), rew, done, info

    def is_action_valid(self, action: np.ndarray) -> bool:
        """Determines if an action is valid or not."""
        if self.action_option == ActionOption.LIFTAFTERROLL:
            (
                duration,
                start_sweeping_pose,
                rolling_distance,
                delta_height,
                delta_yaw,
                delta_pitch,
            ) = np.split(action, [1, 6, 7, 8, 9])
            if duration[0] <= 0:
                return False
            start_pose = sapien.Pose(
                p=start_sweeping_pose[:3],
                q=transforms3d.euler.euler2quat(
                    0, start_sweeping_pose[4], start_sweeping_pose[3]
                ),
            )
            if not self.is_pin_above_table(start_pose):
                return False
            end_pose = self._compute_step_relative_pose(
                start_pose,
                rolling_distance[0],
                delta_height[0],
                delta_yaw[0],
                delta_pitch[0],
            )
            if not self.is_pin_above_table(end_pose):
                return False
            return True
        elif self.action_option == ActionOption.RELATIVE:
            duration, rolling_distance, delta_height, delta_yaw, delta_pitch = action
            if duration <= 0:
                return False
            start_pose = self.agent.robot.get_pose()
            end_pose = self._compute_step_relative_pose(
                start_pose, rolling_distance, delta_height, delta_yaw, delta_pitch
            )
            if not self.is_pin_above_table(end_pose):
                return False
            return True
        elif self.action_option == ActionOption.ABSOLUTE:
            duration, command_pin_position, command_pin_quaternion = np.split(
                action, [1, 4]
            )
            if duration[0] <= 0:
                return False
            if self.is_pin_above_table(
                sapien.Pose(p=command_pin_position, q=command_pin_quaternion)
            ):
                return False
            return True
        else:
            raise NotImplementedError

    def dough_center(self) -> np.ndarray:
        """Return the average position of all particles in the dough."""
        particle_q = self.copy_array_to_numpy(
            self.mpm_states[0].struct.particle_q, self.mpm_model.struct.n_particles
        )
        return np.mean(particle_q, axis=0)

    def _get_obs(self):
        # TODO(blake.wulfe): Generalize this to different obs modes.
        return self.calc_heightmap(
            self._obs_height_map_dx,
            self._obs_height_map_grid_size,
        )
