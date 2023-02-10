import collections
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from ray.rllib.policy.sample_batch import SampleBatch
import torch

from mani_skill2.data.rllib_data_utils import load_sample_batches


class SampleBatchDataset(torch.utils.data.Dataset):
    """A torch dataset that converts from SampleBatch data.

    This is useful when data has been collected using rllib, but you want to use that
    data to train a model using the pytorch ecosystem of tools.

    TOOD(blake.wulfe): Change the implementation such that it doesn't just load everything into memory.
    This is actually easy to do using IterableDataset.

    Args:
        inputs: List of files or file patterns from which to load the sample batches.
            This is the same as rllib's JsonReader input.
        keys: The keys from the stored `SampleBatch`s to return in the sample.
        non_error_keys: Additional keys to extract, but if they don't exist do not result in an error.
        state_transform: Callable to apply to the values associated with state keys.
        debug_size: If provided, limit the size of the dataset to this amount.
            This is not exact, but instead loads the minimum number of saved batches to surpass this value.
        debug_size_mode: Mode for loading debug_size. See `load_sample_batches` documentation.
    """

    # The keys associated with states.
    STATE_KEYS = (SampleBatch.OBS, SampleBatch.NEXT_OBS)
    # The key associated with rewards.
    REWARD_KEY = SampleBatch.REWARDS
    # The key associated with actions.
    ACTION_KEY = SampleBatch.ACTIONS

    def __init__(
        self,
        inputs: List[str],
        keys: Tuple[str] = (
            SampleBatch.OBS,
            SampleBatch.ACTIONS,
            SampleBatch.NEXT_OBS,
            SampleBatch.DONES,
            SampleBatch.REWARDS,
        ),
        non_error_keys: Tuple[str] = (SampleBatch.INFOS,),
        state_transform: Optional[Callable] = None,
        debug_size: Optional[int] = None,
        debug_size_mode: str = "ordered",
    ):
        self.batch = load_sample_batches(inputs, debug_size, debug_size_mode)
        self.keys = keys
        self.non_error_keys = non_error_keys
        self.length = len(self.batch)
        self.state_transform = state_transform

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index) -> Dict:
        sample = dict()
        for key in self.keys + self.non_error_keys:
            # Extract the value associated with the key.
            try:
                value = self.batch[key][index]
            except KeyError as e:
                if key in self.non_error_keys:
                    continue
                else:
                    raise e

            # Format that value.
            if key in self.STATE_KEYS and self.state_transform is not None:
                value = self.state_transform(value)
            if key == self.REWARD_KEY:
                value = np.float32(value)
            if key == self.ACTION_KEY and value.dtype == np.float64:
                value = np.float32(value)
            if not isinstance(value, collections.Mapping):
                # Values should always be batched as 2d tensors.
                # To ensure this, each individual value should at least be 1d.
                value = np.atleast_1d(value)

            # Set the value in the sample.
            sample[key] = value
        return sample
