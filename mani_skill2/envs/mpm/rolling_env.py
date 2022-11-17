from collections import OrderedDict

import h5py
import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat

from mani_skill2 import ASSET_DIR
from mani_skill2.agents.configs.panda.variants import PandaPinchConfig

from mani_skill2.agents.robots.panda import Panda
#from mani_skill2.agents.robots.rolling_pin import RollingPin
from mani_skill2.envs.mpm.base_env import MPMBaseEnv, MPMModelBuilder, MPMSimulator
from mani_skill2.utils.registration import register_gym_env
from mani_skill2.utils.sapien_utils import get_entity_by_name, vectorize_pose

import warp as wp


def generate_circular_cone_heightmap(radius: float, height: float, dx: float)-> np.ndarray:
    half_width = int(radius / dx)
    width = 2 * half_width + 1
    height_map = np.zeros((width, width), dtype=np.float32)
    X, Y = np.meshgrid(np.linspace(-half_width * dx, half_width * dx, width), np.linspace(-half_width * dx, half_width * dx, width))
    height_map = height - np.sqrt(X ** 2 + Y ** 2) / radius * height
    height_map = np.clip(height_map, a_min=np.zeros_like(height_map), a_max=None)
    return height_map

@register_gym_env("Rolling-v0", max_episode_steps=10000)
class RollingEnv(MPMBaseEnv):

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

    def reset(self, *args, seed=None, **kwargs):
        return super().reset(*args, seed=seed, **kwargs)

    def _initialize_mpm(self):
        self.model_builder.clear_particles()

        # I might need to tune the Young's modulus later.
        E = 1E5
        nu = 0.3
        mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
        ys = 2000.
        # 0 for von-mises. In Pinch-v0 it also uses type=0
        type = 0
        friction_angle = 0.5
        cohesion = 0.05
        # An arbitrary initial heightmap.
        dx = 0.002
        height_map = generate_circular_cone_heightmap(radius=0.1, height=0.04, dx=dx)

        count = self.model_builder.add_mpm_from_height_map(
            pos=(0., 0., 0.),
            vel=(0., 0., 0.),
            dx=dx,
            height_map=height_map,
            density=1.2E3,
            mu_lambda_ys=(mu, lam, ys),
            friction_cohesion=(friction_angle, cohesion, 0.),
            type=type,
            jitter=False,
            color=(1., 0., 0.))

        self.model_builder.init_model_state(self.mpm_model, self.mpm_states)

        # I copied these values from pinch_env.py
        self.mpm_model.struct.static_ke = 100.0
        self.mpm_model.struct.static_kd = 0.5
        self.mpm_model.struct.static_mu = 0.9
        self.mpm_model.struct.static_ka = 0.0

        self.mpm_model.struct.body_ke = 100.0
        self.mpm_model.struct.body_kd = 0.2
        self.mpm_model.struct.body_mu = 0.5
        self.mpm_model.struct.body_ka = 0.0

        self.mpm_model.adaptive_grid = False

        self.mpm_model.grid_contact = False
        self.mpm_model.particle_contact = True
        self.mpm_model.struct.body_sticky = False
        self.mpm_model.struct.ground_sticky = 1

        self.mpm_model.struct.particle_radius = 0.0025

    def _get_coupling_actors(self):
        return [
            l for l in self.agent.robot.get_links() if l.name in
            ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
        ]

    def _load_agent(self):
        self.agent = Panda(
            self._scene,
            self.control_freq,
            control_mode=self._control_mode,
            config=PandaPinchConfig(),
        )
        self.grasp_site: sapien.Link = get_entity_by_name(
            self.agent.robot.get_links(), "panda_hand_tcp")

    def _initialize_agent(self):
        noise = self._episode_rng.uniform([-0.1] * 7 + [0, 0],
                                          [0.1] * 7 + [0, 0])
        qpos = np.array([0, 0.01, 0, -1.96, 0.0, 1.98, 0.0, 0.06, 0.06
                         ]) + noise

        self.agent.reset(qpos)
        self.agent.robot.set_pose(sapien.Pose([-0.56, 0, 0]))

    def step(self, *args, **kwargs):
        self._chamfer_dist = None
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
