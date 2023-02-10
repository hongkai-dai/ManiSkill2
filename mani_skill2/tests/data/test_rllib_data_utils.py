import numpy as np
from ray.rllib.policy.sample_batch import SampleBatch

from mani_skill2.data.rllib_data_utils import concat_overlapping_keys_of_samples


class TestConcatOverlappingKeysOfSamples:
    # pylint: disable=redefined-outer-name
    def test_concat_overlapping_keys_of_samples(self):
        samples_1 = SampleBatch(
            {
                SampleBatch.OBS: np.array([1, 2]),
                SampleBatch.ACTIONS: np.array([1, 2]),
            }
        )
        samples_2 = SampleBatch(
            {
                SampleBatch.OBS: np.array([3, 4]),
            }
        )

        concat = concat_overlapping_keys_of_samples([samples_1, samples_2])

        assert SampleBatch.OBS in concat
        assert SampleBatch.ACTIONS not in concat
        np.testing.assert_array_equal(concat[SampleBatch.OBS], [1, 2, 3, 4])
