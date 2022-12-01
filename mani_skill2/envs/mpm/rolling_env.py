from collections import OrderedDict

import h5py
import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat

from mani_skill2 import ASSET_DIR
from mani_skill2.agents.configs.panda.variants import PandaPinchConfig
from mani_skill2.agents.configs.rolling_pin.defaults import RollingPinDefaultConfig

from mani_skill2.agents.robots.panda import Panda
from mani_skill2.agents.robots.rolling_pin import RollingPin
from mani_skill2.envs.mpm.base_env import MPMBaseEnv, MPMModelBuilder, MPMSimulator
from mani_skill2.utils.registration import register_gym_env
from mani_skill2.utils.sapien_utils import get_entity_by_name, vectorize_pose

import warp as wp


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


@register_gym_env("Rolling-v0", max_episode_steps=10000)
class RollingEnv(MPMBaseEnv):
    agent: RollingPin

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
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

    def _initialize_mpm(self):
        self.model_builder.clear_particles()

        # I might need to tune the Young's modulus later.
        E = 2E5
        nu = 0.1
        mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
        ys = 1e3
        # 0 for von-mises. In Pinch-v0 it also uses type=0
        type = 0
        friction_angle = 0.5
        cohesion = 0.05
        dx = 0.0025
        # An arbitrary initial heightmap.
        height_map = generate_circular_cone_heightmap(radius=0.1,
                                                      height=0.06,
                                                      dx=dx)

        count = self.model_builder.add_mpm_from_height_map(
            pos=(0., 0., 0.),
            vel=(0., 0., 0.),
            dx=dx,
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
        qpos = np.array([0, 0., 0.1, np.sqrt(2) / 2., 0., 0., np.sqrt(2) / 2])

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
            tfs = [s.ext_body_f.numpy() for s in self.mpm_states[:-1]]
            tfs = np.mean(tfs, 0)

            if np.isnan(tfs).any():
                self.sim_crashed = True
                return

            # Compute the wrench from applied by the controller.
            if self.mpm_states[-1].struct.error.numpy()[0] == 1:
                self.sim_crashed = True
                return

            controller_wrench = self.agent.controller.compute_wrench(
                scene_time, self.agent.robot.get_pose(),
                self.agent.robot.get_velocity(),
                self.agent.robot.get_angular_velocity())
            for actor, tf in zip(self._coupled_actors, tfs):
                if actor.type not in ["kinematic", "static"]:
                    if actor is self.agent.robot:
                        actor.add_force_torque(tf[3:] + controller_wrench[:3],
                                               tf[:3] + controller_wrench[3:6])
                    else:
                        actor.add_force_torque(tf[3:], tf[:3])

            self._scene.step()
            self.mpm_states = [self.mpm_states[-1]
                               ] + self.mpm_states[:-1]  # rotate states
