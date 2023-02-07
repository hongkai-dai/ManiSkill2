"""Functions relating to generating rollouts.

TODO(blake.wulfe): Switch to some existing framework
instead of rolling your own here.
"""

import collections

import numpy as np


def rollout_episode(env, agent, max_steps):
    """Rolls out an episode in env with the agent.

    Args:
        env: Environment to roll out. Gym.Env interface expected.
        agent: Agent to generate actions.
        max_steps: Maximum number of steps to take in env.

    Returns:
        Dict with lists for relevant string keys.
    """
    traj = collections.defaultdict(list)

    def add_step(**kwargs):
        for k, v in kwargs.items():
            traj[k].append(v)

    agent.reset()
    obs = env.reset()
    for t in range(max_steps):
        action = agent.step(obs)
        next_obs, rew, done, info = env.step(action)

        add_step(
            obs=obs,
            action=action,
            reward=rew,
            next_obs=next_obs,
            done=done,
            trunc=False,
            info=info,
        )

        if done:
            break
        obs = next_obs
    else:
        # Runs at the end of the for loop if there wasn't a break.
        assert len(traj["trunc"]) > 0
        traj["trunc"][-1] = True

    assert len(traj["obs"]) > 0
    return traj


def generate_rollouts(env, agent, num_episodes, max_steps):
    trajs = []
    for i in range(num_episodes):
        traj = rollout_episode(env, agent, max_steps)
        trajs.append(traj)
    return trajs


def convert_rollouts_to_tensor_dict(trajs):
    stacked = collections.defaultdict(list)
    for traj in trajs:
        for k, v in traj.items():
            stacked[k].extend(v)

    tensors = dict()
    for k, v in stacked.items():
        tensors[k] = np.array(v)

    return tensors


def save_rollouts(filepath, trajs):
    tensors = convert_rollouts_to_tensor_dict(trajs)
    np.savez(filepath, **tensors)


def load_rollouts(filepath):
    tensors = np.load(filepath, allow_pickle=True)
    return dict(tensors)
