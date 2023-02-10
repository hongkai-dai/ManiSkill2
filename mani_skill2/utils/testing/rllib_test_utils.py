import gym
from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.policy.sample_batch import SampleBatch


def random_sample_batch_generator(
    obs_space: gym.spaces.Space,
    act_space: gym.spaces.Space,
    num_batches: int,
    batch_size: int,
) -> SampleBatch:
    """Generates random SampleBatch instances.

    args:
        obs_space: Observation space from which to sample obs.
        act_space: Action space from which to sample actions.
        num_batches: The number of batches to generate.
        batch_size: The size of each batch.eeee

    Yields:
        A `SampleBatch` instace with random data.
    """
    for _ in range(num_batches):
        obs = [obs_space.sample() for _ in range(batch_size)]
        act = [act_space.sample() for _ in range(batch_size)]
        next_obs = [obs_space.sample() for _ in range(batch_size)]
        rewards = [0 for _ in range(batch_size)]
        dones = [False for _ in range(batch_size)]
        infos = [dict(a=[1, 2], b=[3, 4]) for _ in range(batch_size)]
        batch = SampleBatch(
            {
                SampleBatch.OBS: obs,
                SampleBatch.ACTIONS: act,
                SampleBatch.NEXT_OBS: next_obs,
                SampleBatch.REWARDS: rewards,
                SampleBatch.DONES: dones,
                SampleBatch.INFOS: infos,
            }
        )
        yield batch


def write_random_sample_batches_to_json(
    dirpath: str,
    obs_space: gym.spaces.Space,
    act_space: gym.spaces.Space,
    num_samples: int,
) -> None:
    """Writes random sample batches to json.

    args:
        dirpath: Directory in which to write the samples.
        obs_space: Observation space from which to sample obs.
        act_space: Action space from which to sample actions.
        num_samples: The number of samples (not batches) to write.
    """
    # Create a writer that will write each batch in its own file.
    writer = JsonWriter(dirpath, max_file_size=8)
    # Write each batch with one sample.
    for batch in random_sample_batch_generator(
        obs_space, act_space, num_batches=num_samples, batch_size=1
    ):
        writer.write(batch)
