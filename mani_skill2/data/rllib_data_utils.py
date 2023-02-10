from typing import List, Optional

from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.policy.sample_batch import SampleBatch


def concat_overlapping_keys_of_samples(samples: List[SampleBatch]) -> SampleBatch:
    """Custom version of SampleBatch.concat_samples that only takes the overlapping keys.

    Args:
        samples: The list of samples to concat.

    Returns:
        A sample batch with the samples concatenated, but only consisting of the common keys.
    """
    assert len(samples) > 0
    assert isinstance(samples[0], SampleBatch)
    overlapping_keys = set(samples[0].keys())
    for batch in samples[1:]:
        overlapping_keys = overlapping_keys.intersection(batch.keys())

    for batch in samples:
        batch_keys = list(batch.keys())
        for key in batch_keys:
            if key not in overlapping_keys:
                del batch[key]

    return SampleBatch.concat_samples(samples)


def load_sample_batches(
    inputs: List[str],
    debug_size: Optional[int] = None,
    debug_size_mode: str = "ordered",
    only_overlapping_keys: bool = True,
) -> SampleBatch:
    """Loads sample batches from inputs into memory and concatenates them into a single `SampleBatch`.

    Args:
        inputs: List of input filepath patterns. Same as what rllib.offline.json_reader.JsonReader takes.
        debug_size: If provided, limits the number of samples to this number.
        debug_size_mode: Mode for loading debug_size.
            "ordered": loads the first debug_size elements, which means they are likely correlated.
                This mode should be used when you want to quickly load the data.
            "shuffled": loads all the data then randomly selects debug_size elements.
                This mode should be used when you want to load a limited amount of data, but in a minimally
                correlated fashion.
        only_overlapping_keys: If True, only selects overlapping keys. If False, raising an error
            when keys are encountered that differ across samples.

    Returns:
        A SampleBatch containing the data loaded into memory.
    """
    concat_fn = (
        concat_overlapping_keys_of_samples
        if only_overlapping_keys
        else SampleBatch.concat_samples
    )

    reader = JsonReader(inputs)
    if debug_size is None:
        batches = list(reader.read_all_files())
        return concat_fn(batches)
    else:
        num_samples = 0
        batches = []
        for batch in reader.read_all_files():
            batches.append(batch)
            num_samples += len(batch)
            if num_samples >= debug_size and debug_size_mode == "ordered":
                break
        samples = concat_fn(batches)
        if debug_size_mode == "shuffled":
            samples.shuffle()
        samples = samples.slice(0, debug_size)
        return samples
