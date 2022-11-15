from collections import OrderedDict
from typing import Dict
import numpy as np
import sapien.core as sapien

from gym import spaces

from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.agents.camera import MountedCameraConfig, get_camera_images, get_camera_pcd


class RollingPinPositionController:
    """ An inverse dynamics controller that only controls the position.

    u_force = -mg + m * Kp(pos_des - pos_curr) - m * Kd * vel_curr
    u_torque = 0.
    This is a "placeholder" controller to test the API.
    """
    pin: sapien.Actor
    gravity: np.ndarray
    Kp: float
    Kd: float

    def __init__(self,
                 pin: sapien.Actor,
                 gravity: np.ndarray = np.array([0, 0, -9.81]),
                 Kp: float = 1.,
                 Kd: float = 2.):
        self.pin = pin
        self.gravity = gravity
        self._pos_des = np.array([0., 0., 0.])
        self.Kp = Kp
        self.Kd = Kd

    def set_goal(self, goal: np.ndarray):
        """
        Args:
          goal The desired position of the pin (no orientation)
        """
        self._pos_des = goal

    def compute_wrench(self, scene_time: float, pose_curr: sapien.Pose,
                       velocity: np.ndarray,
                       angular_velocity: np.ndarray) -> np.ndarray:
        """
        Compute the force/torque expressed in the world frame.
        """
        # Some inputs are not used, but I still leave them in the
        # argument to stay consistent with other controllers.
        mass = self.pin.get_mass()
        force = -mass * self.gravity + mass * self.Kp * (
            self._pos_des - pose_curr.p) + mass * self.Kd * -velocity
        torque = np.array([0., 0., 0.])
        return np.concatenate((force, torque))


class RollingPinInverseDynamicsController:
    """
    A simple inverse dynamics controller for the rolling pin
    u_force = -mg + kp * (pos_des - pos) + kd * (vel_des - vel)
    u_torque = I_world * (kp * orient_error(quat_des, quat) + kd * (omega_des - omega))
    Note that all quantities are expressed in the world frame.
    """
    # TODO(hongkai.dai): add an abstract class RollingPinController to define
    # the common interface.
    pass


class RollingPinConfig:
    """ Configuration for RollingPin used in the MPMEnv.
    """

    def __init__(self):
        pass

    @property
    def cameras(self):
        return dict()


class RollingPin:
    """
    I mimic RollingPin as a BaseAgent. But BaseAgent requires the robot to be sapien.Articulation.
    Our robot (rolling pin) is just a single rigid body.
    """
    scene: sapien.Scene
    robot: sapien.Actor
    cameras: Dict[str, sapien.CameraEntity]
    controller: RollingPinPositionController

    def __init__(
        self,
        scene: sapien.Scene,
        control_freq: int,
    ):
        self.scene = scene
        self._control_freq = control_freq
        self._config = self.get_default_config()
        self.camera_configs = self._config.cameras

        self._setup_actor()
        self._setup_controller()
        self._setup_cameras()

    def _setup_actor(self):
        # Build the rolling pin actor
        actor_builder: sapien.ActorBuilder = self.scene.create_actor_builder()
        capsule_radius = 0.03
        capsule_half_length = 0.2
        material = self.scene.create_physical_material(static_friction=1.,
                                                       dynamic_friction=0.9,
                                                       restitution=0.)
        actor_builder.add_capsule_collision(pose=sapien.Pose(p=[0, 0, 0],
                                                             q=[1, 0, 0, 0]),
                                            radius=capsule_radius,
                                            half_length=capsule_half_length,
                                            material=material,
                                            density=1E3,
                                            patch_radius=0.1,
                                            min_patch_radius=0.1)
        actor_builder.add_capsule_visual(pose=sapien.Pose(p=[0., 0, 0],
                                                          q=[1, 0, 0, 0]),
                                         radius=capsule_radius,
                                         half_length=capsule_half_length,
                                         color=[0., 1, 0],
                                         name="pin")
        self.robot = actor_builder.build(name="pin")

    def _setup_controller(self):
        self.controller = RollingPinPositionController(pin=self.robot,
                                                       gravity=np.array(
                                                           [0, 0, -9.81]),
                                                       Kp=900.,
                                                       Kd=60.)

    def _setup_cameras(self):
        self.cameras = OrderedDict()
        #for uuid, config in self.camera_configs.items():
        #    self.cameras[uuid] = self.scene.add_camera(
        #        name=uuid, width=config.width, height=config.height, fovy = 0, near=config.near, far=config.far)

    def get_default_config(self):
        return RollingPinConfig()

    @property
    def action_space(self):
        return spaces.Box(low=np.array([-1.] * 6), high=np.array([1.] * 6))

    def reset(self, init_qpos=None):
        """
        init_qpos is [px, py, pz, quat_x, quat_y, quat_z, quat_w]
        """
        if init_qpos is not None:
            init_pose = sapien.Pose(
                p=init_qpos[:3],
                q=[init_qpos[6], init_qpos[3], init_qpos[4], init_qpos[5]])
            self.robot.set_pose(init_pose)
        self.robot.set_velocity([0., 0., 0.])
        self.robot.set_angular_velocity([0., 0, 0])

    def before_simulation_step(self):
        pass

    def take_picture(self):
        # I copied this function from BaseAgent.take_picture()
        # NOTE(jigu): take_picture, which starts rendering pipelines, is non-blocking.
        # Thus, calling it before other computation is more efficient.
        for cam in self.cameras.values():
            cam.take_picture()

    def get_camera_images(self,
                          rgb=True,
                          depth=False,
                          visual_seg=False,
                          actor_seg=False) -> Dict[str, Dict[str, np.ndarray]]:
        ret = OrderedDict()
        return ret

    def get_camera_poses(self) -> Dict[str, np.ndarray]:
        poses = OrderedDict()
        return poses
