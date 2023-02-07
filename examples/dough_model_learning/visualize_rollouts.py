import os

import cv2
import fire
import imageio
import numpy as np

from mani_skill2.utils.rollout import load_rollouts


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

    tensors = load_rollouts(data_filepath)
    # Get the indices where the episodes end
    end_indices = np.where(tensors["trunc"])[0]
    start = 0
    for i, end in enumerate(end_indices):
        filepath = os.path.join(output_dir, f"episode_{i:02d}.gif")
        create_heightmap_transition_gif(
            tensors["obs"][start : end + 1],
            tensors["next_obs"][start : end + 1],
            filepath,
        )
        start = end


if __name__ == "__main__":
    fire.Fire(main)
