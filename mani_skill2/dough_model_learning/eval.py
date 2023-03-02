import logging
import os

import hydra
import numpy as np
from omegaconf import DictConfig
import torch

from mani_skill2.algorithms.action_samplers import (
    RandomDoughRollingActionSampler,
)
from mani_skill2.algorithms.random_shooting import RandomShootingAgent
from mani_skill2.dough_model_learning.dough_reward_models import (
    FlatDoughRollingRewardModel,
)
from mani_skill2.dynamics.generative_env import GenerativeEnv
from mani_skill2.utils.rollout import (
    generate_rollouts,
    save_sample_batch,
)


# TODO(blake.wulfe): Generalize this and avoid hardcoding.
# TODO(blake.wulfe): Figure out if you can use hydra.instantiate for this.
def get_algo(env, cfg):
    module = hydra.utils.instantiate(cfg.module)
    if cfg.eval.checkpoint is not None:
        state_dict = torch.load(cfg.eval.checkpoint)["state_dict"]
        module.load_state_dict(state_dict)
    module.to(cfg.eval.device)
    reward_model = FlatDoughRollingRewardModel()
    generative_env = GenerativeEnv(
        module.dynamics_model,
        reward_model,
        env.observation_space,
        env.action_space,
    )
    # TODO(blake.wulfe): Hardcode as 10 instead of using env.action_space
    # because env.action_space is wrong. Fix that then fix this.
    action_size = 10
    num_samples = 100
    action_sampler = RandomDoughRollingActionSampler(
        action_size,
        num_samples=num_samples,
    )
    algorithm = RandomShootingAgent(
        generative_env,
        action_sampler,
        planning_steps=2,
        discount_factor=0.5,
        verbose_info=False,
    )
    return algorithm


def summarize_rollouts(batch):
    episode_batches = batch.split_by_episode()
    returns = []
    for episode_batch in episode_batches:
        returns.append(episode_batch["rewards"].sum())
    logging.info(f"Mean episode return: {np.mean(returns)}")


@hydra.main(config_path="config", config_name="film_unet")
def main(cfg: DictConfig):
    logging.info(f"Starting evaluation. Output dir: {os.getcwd()}")
    env = hydra.utils.instantiate(cfg.eval.env)

    # algorithm = hydra.utils.instantiate(cfg.eval.algorithm)
    algorithm = get_algo(env, cfg)
    rollouts = generate_rollouts(
        env,
        algorithm,
        cfg.eval.num_episodes,
        cfg.eval.max_num_steps,
        render=True,
    )
    summarize_rollouts(rollouts)
    dirpath = os.getcwd()
    output_filepath = os.path.join(dirpath, "eval_rollouts")
    save_sample_batch(output_filepath, rollouts)
    logging.info("Evaluation complete!")


if __name__ == "__main__":
    main()
