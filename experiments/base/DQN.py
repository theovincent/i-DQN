from tqdm import tqdm
import numpy as np
import jax
from argparse import Namespace

from idqn.sample_collection.exploration import EpsilonGreedySchedule
from idqn.sample_collection.replay_buffer import ReplayBuffer
from idqn.networks.base_q import BaseQ
from idqn.environments.atari import AtariEnv


def train(
    key: jax.random.PRNGKey,
    environment_name: str,
    args: Namespace,
    p: dict,
    q: BaseQ,
    env: AtariEnv,
    replay_buffer: ReplayBuffer,
) -> None:
    experiment_path = f"experiments/{environment_name}/figures/{args.experiment_name}/DQN"

    if args.restart_training:
        first_epoch = p["n_epochs"] // 2
        last_epoch = p["n_epochs"]

        sample_key, exploration_key = np.load(f"{experiment_path}/K_{args.seed}.npy")
        env.load(f"{experiment_path}/E_{args.seed}")
        replay_buffer.load(f"{experiment_path}/R_{args.seed}")
        q.load(f"{experiment_path}/Q_{args.seed}_{first_epoch - 1}")
        n_training_steps = int(np.load(f"{experiment_path}/N_{args.seed}"))

        losses = np.load(f"{experiment_path}/L_{args.seed}.npy")
        average_rewards = np.load(f"{experiment_path}/J_{args.seed}.npy")
    else:
        first_epoch = 0
        last_epoch = p["n_epochs"] // 2

        sample_key, exploration_key = jax.random.split(key)
        n_training_steps = 0
        losses = np.ones((p["n_epochs"], p["n_training_steps_per_epoch"])) * np.nan
        average_rewards = np.ones(p["n_epochs"]) * np.nan

        env.collect_random_samples(sample_key, replay_buffer, p["n_initial_samples"], p["horizon"])

    epsilon_schedule = EpsilonGreedySchedule(
        p["starting_eps"],
        p["ending_eps"],
        p["duration_eps"],
        exploration_key,
        n_training_steps,
    )

    for idx_epoch in tqdm(range(first_epoch, last_epoch)):
        sum_reward = 0
        n_episodes = 0
        idx_training_step = 0
        absorbing = False

        while idx_training_step < p["n_training_steps_per_epoch"] or not absorbing:
            sample_key, key = jax.random.split(sample_key)
            reward, absorbing = env.collect_one_sample(q, q.params, p["horizon"], replay_buffer, epsilon_schedule)

            sum_reward += reward
            n_episodes += int(absorbing)

            losses[
                idx_epoch, np.minimum(idx_training_step, p["n_training_steps_per_epoch"] - 1)
            ] = q.update_online_params(n_training_steps, replay_buffer, key)
            q.update_target_params(n_training_steps)

            idx_training_step += 1
            n_training_steps += 1

        average_rewards[idx_epoch] = sum_reward / n_episodes
        np.save(
            f"{experiment_path}/J_{args.seed}.npy",
            average_rewards,
        )
        np.save(
            f"{experiment_path}/L_{args.seed}.npy",
            losses,
        )

    if not args.restart_training:
        np.save(f"{experiment_path}/K_{args.seed}", np.array([sample_key, epsilon_schedule.key]))
        q.save(f"{experiment_path}/Q_{args.seed}_{last_epoch - 1}")
        env.save(f"{experiment_path}/E_{args.seed}")
        replay_buffer.save(f"{experiment_path}/R_{args.seed}")
        np.save(f"{experiment_path}/N_{args.seed}", np.array(n_training_steps))
