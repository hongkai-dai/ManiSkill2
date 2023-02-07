import numpy as np
import pytest

from mani_skill2.utils.rollout import (
    generate_rollouts,
    load_rollouts,
    rollout_episode,
    save_rollouts,
)

EXPECTED_KEYS = ["obs", "action", "reward", "next_obs", "done", "trunc"]


class MockEnv:
    def reset(self):
        return self._get_obs()

    def step(self, action):
        return self._get_obs(), 1, False, {}

    def _get_obs(self):
        return np.zeros((5, 5))


class MockGymAgent:
    def reset(self):
        pass

    def step(self, obs):
        return 0


@pytest.mark.parametrize("max_steps", [1, 2, 10])
def test_rollout_episode(max_steps):
    env = MockEnv()
    agent = MockGymAgent()
    rollout = rollout_episode(env, agent, max_steps=max_steps)
    assert len(rollout["obs"]) == max_steps
    for k in EXPECTED_KEYS:
        assert k in rollout


@pytest.mark.parametrize("num_episodes", [1, 2, 10])
def test_generate_rollouts(num_episodes):
    env = MockEnv()
    agent = MockGymAgent()
    rollouts = generate_rollouts(env, agent, num_episodes=num_episodes, max_steps=5)
    assert len(rollouts) == num_episodes


def test_save_and_load_rollouts(tmp_path):
    env = MockEnv()
    agent = MockGymAgent()
    rollouts = generate_rollouts(env, agent, num_episodes=2, max_steps=5)
    filepath = tmp_path / "test.npz"
    save_rollouts(filepath, rollouts)
    loaded = load_rollouts(filepath)
    for k in EXPECTED_KEYS:
        assert k in loaded
