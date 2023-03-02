"""Functions relating to generating rollouts.

TODO(blake.wulfe): Switch to some existing framework
instead of rolling your own here.
"""
import collections
from typing import Any, Generator

import gym
import numpy as np
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.policy.sample_batch import SampleBatch, concat_samples
from tqdm import tqdm

# TODO(blake.wulfe): Figure out an gym agent interface.
def rollout_episode(
    env: gym.Env,
    agent: Any,
    max_steps: int,
    episode_id: int,
    render: bool = False,
) -> SampleBatch:
    """Rolls out an episode in env with the agent.

    Args:
        env: Environment to roll out. Gym.Env interface expected.
        agent: Agent to generate actions.
        max_steps: Maximum number of steps to take in env.
        episode_id: The int id to use for this episode.

    Returns:
        SampleBatch containing the information from the episode.
    """
    builder = SampleBatchBuilder()
    prep = get_preprocessor(env.observation_space)(env.observation_space)

    agent.reset()
    obs = env.reset()
    rend = None
    if render:
        rend = env.render(mode="rgb_array")
    for t in range(max_steps):
        action, agent_info = agent.step(obs)
        new_obs, rew, done, info = env.step(action)
        info["agent_info"] = agent_info

        builder.add_values(
            t=t,
            eps_id=episode_id,
            obs=prep.transform(obs),
            actions=action,
            rewards=rew,
            new_obs=prep.transform(new_obs),
            dones=done,
            infos=info,
            render=rend,
        )

        if done:
            break
        obs = new_obs
        if render:
            rend = env.render(mode="rgb_array")

    return builder.build_and_reset()


def generate_rollouts(
    env: gym.Env,
    agent: Any,
    num_episodes: int,
    max_steps: int,
    render: bool = False,
) -> SampleBatch:
    """Generates multiple rollouts.

    Args:
        env: Environment to roll out in.
        agent: The agent to use for action selection.
        num_episodes: The number of episodes to generate.
        max_steps: Maximum number of steps per episode.

    Returns:
        A SampleBatch of the concatenated episode rollouts.
    """
    batches = []
    for i in tqdm(range(num_episodes)):
        batch = rollout_episode(env, agent, max_steps, i, render)
        batches.append(batch)
    return concat_samples(batches)


def save_sample_batch(dirpath: str, batch: SampleBatch) -> None:
    """Saves a SampleBatch to file.

    Args:
        dirpath: Directory in which to save individual files.
        batch: The SampleBatch to save.
    """
    writer = JsonWriter(dirpath)
    writer.write(batch)


def load_sample_batch(filepath: str) -> Generator[SampleBatch, None, None]:
    """Load a sample batch from file.

    Args:
        filepath: The filepath to load.

    Returns:
        A generator of SampleBatch objects.
    """
    reader = JsonReader(filepath)
    yield from reader.read_all_files()
