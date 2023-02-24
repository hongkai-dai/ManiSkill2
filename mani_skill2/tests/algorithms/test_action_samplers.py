import pytest
import torch

from mani_skill2.algorithms.action_samplers import (
    RandomDoughRollingActionSampler,
)


class TestRandomDoughRollingActionSampler:
    @pytest.mark.parametrize("num_samples", (1, 2))
    @pytest.mark.parametrize("num_steps", (1, 2))
    @pytest.mark.parametrize("device", ("cuda", "cpu"))
    def test_call(self, num_samples, num_steps, device, action_size=10):
        sampler = RandomDoughRollingActionSampler(
            action_size=action_size,
            num_samples=num_samples,
            device=device,
        )
        actions = sampler(None, num_steps)
        assert actions.shape == (num_samples, num_steps, action_size)
        assert actions.device.type == device
