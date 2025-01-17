"""Generates dough rolling data.

TODO(blake.wulfe): Either improve this or use an existing framework for it.
For now this is just draft code for generating data until we figure out how
we actually want to do it. For example, this should probably use a config.
We should probably plan out our requirements for the paper before doing this.

To run this script (for example):
```
export CUDA_VISIBLE_DEVICES=0
python examples/dough_model_learning/generate_dough_rolling_data.py \
--output_filepath /path/to/where/to/store/data.npz \
--num_episodes=1
```
"""
import functools

import fire

from mani_skill2.dough_model_learning.data_generation_utils import (
    DoughRollingCenterOutAgent,
    DoughRollingRandomActionAgent,
)

from mani_skill2.envs.mpm.rolling_env import (
    RollingEnv,
    generate_circular_cone_heightmap,
)
from mani_skill2.utils.rollout import (
    generate_rollouts,
    save_sample_batch,
)


class CircularConeHeightMapGenerator:
    def __init__(self, radius=0.1, height=0.06):
        self.generate = functools.partial(
            generate_circular_cone_heightmap,
            radius=radius,
            height=height,
        )

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)


def get_env(**kwargs):
    return RollingEnv(**kwargs)


def get_agent(agent_type):
    if agent_type == "random":
        return DoughRollingRandomActionAgent()
    elif agent_type == "out":
        return DoughRollingCenterOutAgent()
    else:
        raise ValueError(agent_type)


def main(
    output_filepath,
    num_episodes,
    max_num_steps=5,
    sim_freq=200,
    mpm_freq=1000,
    obs_height_map_dx=0.01,
    obs_height_map_grid_size=(32, 32),
    agent_type="random",
):
    height_map_generator = CircularConeHeightMapGenerator()
    env = get_env(
        sim_freq=sim_freq,
        mpm_freq=mpm_freq,
        dough_initializer=height_map_generator,
        obs_height_map_dx=obs_height_map_dx,
        obs_height_map_grid_size=obs_height_map_grid_size,
    )
    agent = get_agent(agent_type)
    rollouts = generate_rollouts(
        env,
        agent,
        num_episodes,
        max_num_steps,
        render=False,
    )
    save_sample_batch(output_filepath, rollouts)


if __name__ == "__main__":
    fire.Fire(main)
