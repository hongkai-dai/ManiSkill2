import os

import cv2
import fire
import imageio
import numpy as np

from mani_skill2.utils.rollout import load_sample_batch


def create_gif(imgs, filepath, duration=0.1):
    assert filepath.endswith(".gif")
    imageio.mimsave(filepath, imgs, duration=duration)


def normalize_resize_heightmaps(heightmaps, max_height, resize):
    normalized_heights = [(h / max_height * 255).astype(np.uint8) for h in heightmaps]
    resized_heights = [cv2.resize(h, resize) for h in normalized_heights]
    return resized_heights


def create_heightmap_transition_gif(obs, next_obs, filepath, resize=(256, 256)):
    max_height = np.max(obs)
    obs = normalize_resize_heightmaps(obs, max_height, resize)
    next_obs = normalize_resize_heightmaps(next_obs, max_height, resize)

    transitions = []
    for obs_i, next_obs_i in zip(obs, next_obs):
        transition = np.hstack((obs_i, next_obs_i))
        transitions.append(transition)
    return create_gif(transitions, filepath, duration=0.5)


def main(output_dir, data_filepath, max_num_episodes_to_visalize=5):
    os.makedirs(output_dir, exist_ok=True)

    batch = list(load_sample_batch(data_filepath))[0]
    episodes = batch.split_by_episode()
    for episode in episodes[:max_num_episodes_to_visalize]:
        eps_id = episode["eps_id"][0]
        filepath = os.path.join(output_dir, f"episode_{eps_id:02d}_heightmap.gif")
        create_heightmap_transition_gif(
            episode["obs"],
            episode["new_obs"],
            filepath,
        )
        if "render" in episode and episode["render"][0] is not None:
            filepath = os.path.join(output_dir, f"episode_{eps_id:02d}_render.gif")
            create_gif(episode["render"].astype(np.uint8), filepath, duration=0.5)

    # Compute average return.
    average_return = np.mean([episode["rewards"].sum() for episode in episodes])
    print(f"Average return: {average_return}")


if __name__ == "__main__":
    fire.Fire(main)
