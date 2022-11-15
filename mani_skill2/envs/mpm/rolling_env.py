from collections import OrderedDict

import h5py
import numpy as np
import sapien.core as sapien

from mani_skill2 import ASSET_DIR

from mani_skill2.agents.robots.rolling_pin import RollingPin
from mani_skill2.envs.mpm.base_env import MPMBaseEnv, MPMModelBuilder, MPMSimulator
from mani_skill2.utils.registration import register_gym_env
from mani_skill2.envs.mpm.utils import load_h5_as_dict

import warp as wp


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

    def _setup_mpm(self):
        """
        I copied this function from pinch_env.py
        """
        self.model_builder = MPMModelBuilder()
        self.model_builder.set_mpm_domain(domain_size=[0.5, 0.5, 0.5], grid_length=0.01)
        self.model_builder.reserve_mpm_particles(count=self.max_particles)

        self._setup_mpm_bodies()

        self.mpm_simulator = MPMSimulator(device="cuda")

        height_map = np.array([
            [0., 0., 0., 0., 0., 0.],
            [0., 0.01, 0.03, 0.02, 0.01, 0],
            [0., 0.02, 0.04, 0.03, 0.01, 0],
            [0., 0.02, 0.04, 0.04, 0.02, 0],
            [0., 0.01, 0.015, 0.02, 0.01, 0],
            [0, 0, 0, 0, 0, 0]])
        mu_lambda_ys = [200000., 50000., 20000.]
        friction_cohesion = [0., 0., 0.]
        self.model_builder.add_mpm_from_height_map(pos=[0., 0., 0.02], vel=[0., 0., 0.], dx=0.01, height_map=height_map, density=6E2, mu_lambda_ys=mu_lambda_ys, friction_cohesion=friction_cohesion, type=0)
        self.mpm_model = self.model_builder.finalize(device="cuda")
        # Use a smaller gravity to speed up the simulation.
        self.mpm_model.gravity = np.array((0.0, 0.0, -1), dtype=np.float32)
        self.mpm_model.struct.ground_normal = wp.vec3(0.0, 0.0, 1.0)
        self.mpm_model.struct.particle_radius = 0.005
        self.mpm_states = [self.mpm_model.state() for _ in range(self._mpm_step_per_sapien_step + 1)]

    def _initialize_mpm(self):
        filepath = ASSET_DIR / "pinch/levels/0_A00_0.obj.h5"
        self.info = load_h5_as_dict(h5py.File(str(filepath), "r"))
        n = len(self.info["init_state"]["mpm"]["x"])
        self.mpm_model.struct.n_particles = n
        # sapien state is the state [pos, quat, lin_vel, angular_vel] of the rolling pin.
        sapien_state = np.array([0, 0, 0.1, np.sqrt(2)/2, np.sqrt(2)/2, 0, 0, 0, 0, 0, 0, 0, 0])
        mpm_state = self.info["init_state"]["mpm"]
        state = {"sapien": sapien_state, "mpm": mpm_state}
        self.set_sim_state(state)
        self.mpm_model.mpm_particle_colors = (np.ones((n, 3)) * np.array([0.5, 0.1, 0.2])).astype(np.float32)

        self.mpm_model.struct.static_ke = 100.0
        self.mpm_model.struct.static_kd = 0.5
        self.mpm_model.struct.static_mu = 0.9
        self.mpm_model.struct.static_ka = 0.0

        self.mpm_model.struct.body_ke = 100.0
        self.mpm_model.struct.body_kd = 0.2
        self.mpm_model.struct.body_mu = 0.5
        self.mpm_model.struct.body_ka = 0.0

        self.mpm_model.adaptive_grid = False
        self.mpm_model.particle_contact = True
        self.mpm_model.grid_contact = False
        self.mpm_model.struct.ground_sticky = True
        self.mpm_model.struct.body_sticky = False