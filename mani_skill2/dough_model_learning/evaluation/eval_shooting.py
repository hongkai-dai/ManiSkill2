from typing import Union

import gym
import hydra
import imageio
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import torch

from mani_skill2.algorithms.action_samplers import RandomDoughRollingActionSampler
from mani_skill2.algorithms.gym_agent import GymAgent
from mani_skill2.algorithms.random_shooting import RandomShootingAgent
from mani_skill2.algorithms.gradient_shooting import GradientShootingAgent
from mani_skill2.envs.mpm.rolling_env import RollingEnv
from mani_skill2.dough_model_learning.dough_reward_models import (
    EllipseShape,
    ShapeState,
    ShapeRewardModel,
)
from mani_skill2.dynamics.generative_env import GenerativeEnv, RewardOption
from mani_skill2.dynamics.modules import DynamicsPLModule
from mani_skill2.dough_model_learning.visualize_rollouts import create_gif

from mani_skill2.utils.rollout import (
    generate_rollouts,
)

HEIGHT_MAX = 0.05


def get_dynamics_module(cfg: DictConfig) -> DynamicsPLModule:
    """
    Load the learned dynamics module.
    """
    dynamics_module = hydra.utils.instantiate(cfg.module)
    if cfg.eval.checkpoint is not None:
        state_dict = torch.load(cfg.eval.checkpoint)["state_dict"]
        dynamics_module.load_state_dict(state_dict)
    dynamics_module.to(cfg.eval.device)
    return dynamics_module


def plot_planned_rollout(
    agent: Union[RandomShootingAgent, GradientShootingAgent],
    obs: np.ndarray,
    action_sequence: torch.Tensor,
    figure_name: str,
    output_dir: str,
):
    with torch.no_grad():
        current_state = agent.generative_env.dynamics_model.state_from_observation(
            obs
        ).to(agent.device)

        planning_states, planning_rewards, _ = agent.generative_env.rollout(
            current_state.unsqueeze(0),
            # action_sequences_init[0].unsqueeze(0),
            action_sequence.unsqueeze(0),
            agent.discount_factor,
        )
        rendered_images = []
        for planning_step in range(agent.planning_steps + 1):
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.imshow(
                planning_states[0, planning_step, ...].detach().cpu().numpy(),
                vmax=HEIGHT_MAX,
            )
            ax.set_title(f"Planning step {planning_step}")
            path = f"{output_dir}/{figure_name}{planning_step}.png"
            fig.savefig(path, format="png")
            rendered_images.append(imageio.imread(path))
        imageio.mimsave(f"{output_dir}/{figure_name}.gif", rendered_images, duration=1)
        return planning_states, planning_rewards


def plan_shooting(
    obs: np.ndarray,
    random_agent: RandomShootingAgent,
    gradient_agent: GradientShootingAgent,
    output_dir: str,
):
    """
    First plan withe random shooting to find the best action sequence from a batch of candidates. The use gradient-based shooting to fine tune this action sequence.
    """
    random_next_action, random_agent_info = random_agent.step(obs)
    top_random_rollout_count = 3
    top_random_rollout_indices = torch.topk(
        random_agent_info["total_rewards"], top_random_rollout_count
    ).indices
    for (ranking, index) in enumerate(top_random_rollout_indices):
        plot_planned_rollout(
            random_agent,
            obs,
            random_agent_info["action_sequences"][ranking],
            f"random_shooting_rank{ranking}",
            output_dir,
        )

    gradient_agent.set_action_sequences_init(
        random_agent_info["action_sequences"][top_random_rollout_indices]
    )
    gradient_next_action, gradient_agent_info = gradient_agent.step(obs)
    for rollout_index in range(top_random_rollout_count):
        plot_planned_rollout(
            gradient_agent,
            obs,
            gradient_agent_info["best_act_sequences"][rollout_index],
            f"gradient_shooting_index{rollout_index}",
            output_dir,
        )
    return gradient_next_action, gradient_agent_info


def plan_gradient_shooting(
    obs: np.ndarray, gradient_agent: GradientShootingAgent, output_dir
):
    batch_size = 1
    act_size = 10
    # Manually set the initial action.
    action_sequences_init = torch.empty(
        (batch_size, gradient_agent.planning_steps, act_size),
        device=gradient_agent.device,
    )
    action_sequences_init[0][0][0] = 1  # duration.
    action_sequences_init[0][0][1] = 0  # pos x
    action_sequences_init[0][0][2] = 0  # pos y
    action_sequences_init[0][0][3] = 0.05  # pos z
    action_sequences_init[0][0][4] = 0.2 * np.pi  # yaw
    action_sequences_init[0][0][5] = 0  # pitch
    action_sequences_init[0][0][6] = 0.1  # rolling distance
    action_sequences_init[0][0][7] = 0  # delta height
    action_sequences_init[0][0][8] = 0  # delta yaw
    action_sequences_init[0][0][9] = 0  # delta pitch

    action_sequences_init[0][1][0] = 1  # duration.
    action_sequences_init[0][1][1] = 0  # pos x
    action_sequences_init[0][1][2] = 0  # pos y
    action_sequences_init[0][1][3] = 0.05  # pos z
    action_sequences_init[0][1][4] = 0.7 * np.pi  # yaw
    action_sequences_init[0][1][5] = 0  # pitch
    action_sequences_init[0][1][6] = 0.1  # rolling distance
    action_sequences_init[0][1][7] = 0  # delta height
    action_sequences_init[0][1][8] = 0  # delta yaw
    action_sequences_init[0][1][9] = 0  # delta pitch
    plot_planned_rollout(
        gradient_agent, obs, action_sequences_init[0], "init", output_dir
    )
    gradient_agent.set_action_sequences_init(action_sequences_init.detach())
    action, agent_info = gradient_agent.step(obs)
    plot_planned_rollout(
        gradient_agent,
        obs,
        agent_info["best_act_sequences"][0],
        "gradient_agent",
        output_dir,
    )
    return action, agent_info


def rollout_shooting(
    env: gym.Env,
    gradient_agent: GradientShootingAgent,
    action_sampler: RandomDoughRollingActionSampler,
    max_steps: int,
    render: bool = False,
    output_dir: str = "/tmp",
):
    random_agent = RandomShootingAgent(
        gradient_agent.generative_env,
        action_sampler,
        gradient_agent.planning_steps,
        gradient_agent.discount_factor,
        device=gradient_agent.device,
        verbose_info=False,
    )
    gradient_agent.reset()
    obs = env.reset()
    plot = True
    if plot:
        # Plot the starting heightmap
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(obs, vmax=HEIGHT_MAX)
        ax.set_title("Sim at step 0")
        fig.savefig(f"{output_dir}/dough_t0.png", format="png")

    batch_size = action_sampler.num_samples
    act_size = action_sampler.action_size

    if plot:
        # Plot the desired heightmap
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(
            gradient_agent.generative_env.reward_model.shape.desired_heightmap(
                torch.zeros((1, 2)), torch.tensor(0.0)
            )
            .detach()
            .numpy(),
            vmax=HEIGHT_MAX,
        )
        ax.set_title("desired shape")
        fig.savefig(f"{output_dir}/dough_desired.png", format="png")

    rend = env.render(mode="rgb_array")
    for t in range(max_steps):
        action, agent_info = plan_shooting(
            obs, random_agent, gradient_agent, output_dir
        )
        best_act_sequence = agent_info["best_act_sequences"][
            torch.argmax(agent_info["best_rewards"])
        ]
        new_obs, reward, done, info = env.step(action)
        if plot:
            with torch.no_grad():
                for rollout in range(agent_info["rewards_curve"].shape[1]):
                    fig = plt.figure()
                    ax = fig.add_subplot()
                    ax.plot(
                        agent_info["rewards_curve"][:, rollout].cpu().detach().numpy()
                    )
                    ax.set_title("Reward")
                    fig.savefig(
                        f"{output_dir}/dough_rollout{rollout}_t{t}_reward.png",
                        format="png",
                    )

                fig = plt.figure()
                ax = fig.add_subplot()
                ax.imshow(new_obs, vmax=HEIGHT_MAX)
                ax.set_title(f"Sim at step {t+1}.")
                fig.savefig(
                    f"{output_dir}/dough_t{t+1}.png",
                    format="png",
                )

        if done:
            break
        # create_gif(env.rendered_images, f"{output_dir}/dough_sim.gif", duration=0.05)
        obs = new_obs
        # Shift the previous action sequence by 1, append a random action in the end.
        act_sequence_shift = torch.empty(
            (gradient_agent.planning_steps, act_size), device=gradient_agent.device
        )
        act_sequence_shift[: gradient_agent.planning_steps - 1] = best_act_sequence[
            1:
        ].detach()
        action_sequences_init = act_sequence_shift.repeat(
            [batch_size] + [1] * best_act_sequence.ndim
        )
        # Sample the last action in the prediction horizon.
        action_sequences_init[:, -1, :] = (
            action_sampler(obs, num_steps=1).to(gradient_agent.device).squeeze(1)
        )


@hydra.main(config_path="../config", config_name="film_unet.yaml")
def main(cfg: DictConfig):
    torch.manual_seed(123)
    np.random.seed(123)
    env: RollingEnv = hydra.utils.instantiate(cfg.eval.env)
    heightmap_init = env.calc_heightmap(
        dx=env._obs_height_map_dx, grid_size=env._obs_height_map_grid_size
    )
    dough_volume = heightmap_init.calc_volume()

    dynamics_module = get_dynamics_module(cfg)
    desired_shape = EllipseShape(
        dough_volume,
        #length=0.27,
        #height=0.02,
        length=0.16,
        height=0.025,
        grid_coord_h=torch.from_numpy(heightmap_init.grid_h),
        grid_coord_w=torch.from_numpy(heightmap_init.grid_w),
    )
    reward_model = ShapeRewardModel(desired_shape, ShapeState.Next)
    generative_env = GenerativeEnv(
        dynamics_module.dynamics_model,
        reward_model,
        env.observation_space,
        env.action_space,
        RewardOption.FinalTimeOnly,
    )

    batch_size = 500
    act_size = 10
    action_sampler = RandomDoughRollingActionSampler(
        act_size, batch_size
    )  # , height_bounds=(0.04, 0.1), rolling_distance_bounds=(0.1, 0.4), rolling_duration_bounds=(1.0, 1.0))
    algorithm = GradientShootingAgent(
        generative_env,
        planning_steps=2,
        discount_factor=0.5,
        optimizer=torch.optim.Adam,
        gradient_steps=2000,
        action_bounds=action_sampler.bounds(),
    )

    # GradientShootingAgent requires calling set_action_sequences_init, which is not supported within generate_rollouts. Hence I will not call generate_rollouts here.
    # rollouts = generate_rollouts(generative_env, algorithm, cfg.eval.num_episodes, cfg.eval.max_num_steps)
    rollout_shooting(
        env,
        algorithm,
        action_sampler,
        max_steps=cfg.eval.max_num_steps,
        output_dir=cfg.user.run_dir,
        render=False,
    )


if __name__ == "__main__":
    main()
