import gym
import pytest
from ray.rllib.policy.sample_batch import SampleBatch

from mani_skill2.data.sample_batch_dataset import SampleBatchDataset
from mani_skill2.utils.testing.rllib_test_utils import (
    write_random_sample_batches_to_json,
)


@pytest.fixture
def sample_batches(tmp_path):
    num_samples = 5
    obs_space = gym.spaces.Discrete(4)
    act_space = gym.spaces.Discrete(2)
    write_random_sample_batches_to_json(
        str(tmp_path),
        obs_space,
        act_space,
        num_samples=num_samples,
    )
    return num_samples, obs_space, act_space, tmp_path


class TestSampleBatchDataset:
    # pylint: disable=redefined-outer-name
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_length_and_getitem(self, sample_batches):
        num_samples, obs_space, act_space, tmp_path = sample_batches
        dataset = SampleBatchDataset(str(tmp_path / "*"))
        assert len(dataset) == num_samples
        sample = dataset[0]
        for key in [SampleBatch.OBS, SampleBatch.ACTIONS, SampleBatch.NEXT_OBS]:
            assert key in sample
        assert obs_space.contains(sample[SampleBatch.OBS][0])
        assert act_space.contains(sample[SampleBatch.ACTIONS][0])

    @pytest.mark.parametrize("debug_size", [4, 5, 6])
    # pylint: disable=redefined-outer-name
    def test_debug_size(self, sample_batches, debug_size):
        num_samples, _, _, tmp_path = sample_batches
        dataset = SampleBatchDataset(
            str(tmp_path / "*"), debug_size=debug_size
        )
        expected = num_samples if num_samples < debug_size else debug_size
        assert (
            len(dataset) == expected
        ), "Should equal exactly due to one sample per file."
