import gym
import numpy as np
import pytest

from mani_skill2.utils.rollout import (
    generate_rollouts,
    load_sample_batch,
    rollout_episode,
    save_sample_batch,
)

EXPECTED_KEYS = ["obs", "actions", "rewards", "new_obs", "dones"]


class MockEnv:
    def __init__(
        self,
        observation_space: gym.spaces.Space = gym.spaces.Box(
            low=0,
            high=0,
            shape=(5, 5),
        ),
    ):
        self.observation_space = observation_space

    def reset(self):
        return self._get_obs()

    def step(self, action):
        return self._get_obs(), 1, False, {}

    def _get_obs(self):
        return self.observation_space.sample()


class MockGymAgent:
    def reset(self):
        pass

    def step(self, obs):
        return 0


@pytest.mark.parametrize("max_steps", [1, 2, 10])
def test_rollout_episode(max_steps):
    env = MockEnv()
    agent = MockGymAgent()
    rollout = rollout_episode(
        env,
        agent,
        max_steps=max_steps,
        episode_id=0,
    )
    assert len(rollout["obs"]) == max_steps
    for k in EXPECTED_KEYS:
        assert k in rollout


@pytest.mark.parametrize("num_episodes", [1, 2, 10])
def test_generate_rollouts(num_episodes):
    env = MockEnv()
    agent = MockGymAgent()
    max_steps = 5
    rollouts = generate_rollouts(
        env,
        agent,
        num_episodes=num_episodes,
        max_steps=max_steps,
    )
    assert len(rollouts) == num_episodes * max_steps


class TestSaveAndLoadRollouts:
    @pytest.mark.parametrize(
        "obs_space",
        [
            gym.spaces.Box(low=0, high=0, shape=(5, 5)),
            gym.spaces.Dict(
                {
                    "height": gym.spaces.Box(low=0, high=0, shape=(5, 5)),
                    "grid": gym.spaces.Box(low=0, high=0, shape=(5, 5)),
                }
            ),
        ],
    )
    def test_there_and_back(self, tmp_path, obs_space):
        env = MockEnv(observation_space=obs_space)
        agent = MockGymAgent()
        rollouts = generate_rollouts(env, agent, num_episodes=2, max_steps=5)
        filepath = str(tmp_path / "test.npz")
        save_sample_batch(filepath, rollouts)
        loaded = next(load_sample_batch(filepath))
        for k in EXPECTED_KEYS:
            assert k in loaded
