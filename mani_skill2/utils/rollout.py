"""Functions relating to generating rollouts.

TODO(blake.wulfe): Switch to some existing framework
instead of rolling your own here.
"""

import collections

import numpy as np
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.policy.sample_batch import SampleBatch


def rollout_episode(env, agent, max_steps, episode_id):
    """Rolls out an episode in env with the agent.

    Args:
        env: Environment to roll out. Gym.Env interface expected.
        agent: Agent to generate actions.
        max_steps: Maximum number of steps to take in env.

    Returns:
        Dict with lists for relevant string keys.
    """
    builder = SampleBatchBuilder()
    prep = get_preprocessor(env.observation_space)(env.observation_space)

    agent.reset()
    obs = env.reset()
    for t in range(max_steps):
        action = agent.step(obs)
        new_obs, rew, done, info = env.step(action)

        builder.add_values(
            t=t,
            eps_id=episode_id,
            obs=prep.transform(obs),
            actions=action,
            rewards=rew,
            new_obs=prep.transform(new_obs),
            dones=done,
            infos=info,
        )

        if done:
            break
        obs = new_obs

    return builder.build_and_reset()


def generate_rollouts(env, agent, num_episodes, max_steps):
    batches = []
    for i in range(num_episodes):
        batch = rollout_episode(env, agent, max_steps, i)
        batches.append(batch)
    return SampleBatch.concat_samples(batches)


def save_sample_batch(filepath, batch):
    writer = JsonWriter(filepath)
    writer.write(batch)


def load_sample_batch(filepath):
    reader = JsonReader(filepath)
    yield from reader.read_all_files()
