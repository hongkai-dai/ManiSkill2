import gym
import numpy as np

from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.visualization.cv2_utils import OpenCVViewer
from mani_skill2.envs.mpm.rolling_env import RollingEnv

def main():
    env: BaseEnv = gym.make("Rolling-v0", control_mode="pd_ee_delta_pose")

    obs = env.reset()
    after_reset = True

    env.render(mode="human")
    opencv_viewer = OpenCVViewer()

    def render_wait():
        while True:
            sapien_viewer = env.render(mode="human")
            if sapien_viewer.window.key_down("0"):
                break

    has_gripper = "gripper" in env.agent.controller.configs
    gripper_action = 1
    EE_ACTION = 0.1

    while True:
        env.render(mode="human")

        render_frame = env.render(mode="cameras")

        if after_reset:
            after_reset = False
            # Re-focus on opencv viewer
            opencv_viewer.close()
            opencv_viewer = OpenCVViewer()

        key = opencv_viewer.imshow(render_frame)

        ee_action = np.zeros([6])
        # Position
        if key == "i":  # +x
            ee_action[0] = EE_ACTION
        elif key == "k":  # -x
            ee_action[0] = -EE_ACTION
        elif key == "j":  # +y
            ee_action[1] = EE_ACTION
        elif key == "l":  # -y
            ee_action[1] = -EE_ACTION
        elif key == "u":  # +z
            ee_action[2] = EE_ACTION
        elif key == "o":  # -z
            ee_action[2] = -EE_ACTION

        if key == "q":
            break

    gripper_action = 1  # open gripper
    action = np.hstack([ee_action, gripper_action])
    env.step_action(action)
    #obs = env.get_obs()
    #info = env.get_info(obs=obs)
    #done = env.get_done(obs=obs, info=info)

if __name__ == "__main__":
    main()