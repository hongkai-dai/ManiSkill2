from copy import deepcopy
import numpy as np
import transforms3d
import typing

from mani_skill2.envs.mpm.rolling_env import RollingEnv
from mani_skill2.utils.visualization.cv2_utils import OpenCVViewer


def key_to_pose(key: str, position: np.ndarray, quaternion: np.ndarray,
                R_current: np.ndarray, EE_ACTION: float,
                delta_theta: float) -> typing.Tuple[np.ndarray, np.ndarray]:
    position_new = deepcopy(position)
    quaternion_new = deepcopy(quaternion)
    if key == "i":  # move forward
        position_new += R_current @ np.array([0, EE_ACTION, 0])
    elif key == "k":  # move backward
        position_new -= R_current @ np.array([0, EE_ACTION, 0])
    elif key == "j":  # side step
        position_new += R_current @ np.array([EE_ACTION, 0, 0])
    elif key == "l":  # side step
        position_new -= R_current @ np.array([EE_ACTION, 0, 0])
    elif key == "u":  # +z
        position_new[2] += EE_ACTION
    elif key == "o":  # -z
        position_new[2] += -EE_ACTION
    elif key == "y":  # + yaw
        # yaw in the body frame.
        quaternion_new = transforms3d.quaternions.qmult(
            quaternion,
            np.array([np.cos(delta_theta / 2), 0, 0,
                      np.sin(delta_theta / 2)]))
    elif key == "h":  # - yaw
        # yaw in the body frame
        quaternion_new = transforms3d.quaternions.qmult(
            quaternion,
            np.array(
                [np.cos(-delta_theta / 2), 0, 0,
                 np.sin(-delta_theta / 2)]))
    elif key == "p":  # + pitch
        # pitch in the body frame.
        quaternion_new = transforms3d.quaternions.qmult(
            quaternion,
            np.array([np.cos(delta_theta / 2), 0,
                      np.sin(delta_theta / 2), 0]))
    elif key == ";":  # - pitch
        # pitch in the body frame.
        quaternion_new = transforms3d.quaternions.qmult(
            quaternion,
            np.array(
                [np.cos(-delta_theta / 2), 0,
                 np.sin(-delta_theta / 2), 0]))
    return position_new, quaternion_new


def main():
    SIM_FREQ = 500
    MPM_FREQ = 2000
    env = RollingEnv(sim_freq=SIM_FREQ, mpm_freq=MPM_FREQ)
    control_dt = 1 / env.control_freq

    obs = env.reset()
    after_reset = True

    env.render(mode="human")
    opencv_viewer = OpenCVViewer()

    EE_ACTION = 0.01

    sim_step = 0
    sim_time = 0.
    while True:
        env.render(mode="human")

        render_frame = env.render(mode="cameras")

        if after_reset:
            after_reset = False
            # Re-focus on opencv viewer.
            opencv_viewer.close()
            opencv_viewer = OpenCVViewer()

        key = opencv_viewer.imshow(render_frame)

        pose = env.agent.robot.get_pose()
        position = pose.p
        quaternion = pose.q
        R_current = transforms3d.quaternions.quat2mat(quaternion)
        delta_theta = 5. / 180 * np.pi
        print(f"current_pose={pose}")

        position_new, quaternion_new = key_to_pose(key, position, quaternion,
                                                   R_current, EE_ACTION,
                                                   delta_theta)

        if key == "q":
            break
        pose.set_p(position_new)
        pose.set_q(quaternion_new)
        print(f"desired pose={pose}")
        env.step_action_one_way_coupling(pose)
        sim_step += 1
        sim_time += control_dt
        #obs = env.get_obs()
        #info = env.get_info(obs=obs)
        #done = env.get_done(obs=obs, info=info)


if __name__ == "__main__":
    main()
