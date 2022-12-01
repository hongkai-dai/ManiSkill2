import gym
import numpy as np

from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.visualization.cv2_utils import OpenCVViewer
from mani_skill2.envs.mpm.rolling_env import RollingEnv


def main():
    #env: BaseEnv = gym.make("Rolling-v0", control_mode="pd_ee_delta_pose")
    sim_freq = 500
    mpm_freq = 2000
    env = RollingEnv(sim_freq=sim_freq, mpm_freq=mpm_freq)
    control_dt = 1 / env.control_freq

    obs = env.reset()
    after_reset = True

    env.render(mode="human")
    opencv_viewer = OpenCVViewer()

    def render_wait():
        while True:
            sapien_viewer = env.render(mode="human")
            if sapien_viewer.window.key_down("0"):
                break

    EE_ACTION = 0.01

    sim_step = 0
    sim_time = 0.
    while True:
        env.render(mode="human")

        render_frame = env.render(mode="cameras")

        if after_reset:
            after_reset = False
            # Re-focus on opencv viewer
            opencv_viewer.close()
            opencv_viewer = OpenCVViewer()

        key = opencv_viewer.imshow(render_frame)

        action = env.agent.robot.get_pose().p
        # Position
        if key == "i":  # +x
            action[0] += EE_ACTION
        elif key == "k":  # -x
            action[0] += -EE_ACTION
        elif key == "j":  # +y
            action[1] += EE_ACTION
        elif key == "l":  # -y
            action[1] += -EE_ACTION
        elif key == "u":  # +z
            action[2] += EE_ACTION
        elif key == "o":  # -z
            action[2] += -EE_ACTION

        if key == "q":
            break
        env.step_action(sim_time, action)
        sim_step += 1
        sim_time += control_dt
        #obs = env.get_obs()
        #info = env.get_info(obs=obs)
        #done = env.get_done(obs=obs, info=info)


if __name__ == "__main__":
    main()
