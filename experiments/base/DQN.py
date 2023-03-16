from tqdm import tqdm
import numpy as np
import jax
from argparse import Namespace

from idqn.sample_collection.exploration import EpsilonGreedySchedule
from idqn.sample_collection.replay_buffer import ReplayBuffer
from idqn.networks.base_q import BaseQ
from idqn.utils.pickle import load_pickled_data, save_pickled_data
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
        first_epoch = p["n_epochs"] // 2 + 1
        last_epoch = p["n_epochs"]

        sample_key, exploration_key = np.load(f"{experiment_path}/K_{args.seed}.npy")
        env.load(f"{experiment_path}/E_{args.seed}")
        replay_buffer.load(f"{experiment_path}/R_{args.seed}")
        q.params = load_pickled_data(f"{experiment_path}/P_{args.seed}_{first_epoch - 1}", device_put=True)
        q.target_params = load_pickled_data(f"{experiment_path}/TP_{args.seed}_{first_epoch - 1}", device_put=True)
        q.optimizer_state = load_pickled_data(f"{experiment_path}/O_{args.seed}", device_put=True)
        n_gradient_steps = int(np.load(f"{experiment_path}/N_{args.seed}"))

        l2_losses = np.load(f"{experiment_path}/L_{args.seed}.npy")
        average_rewards = np.load(f"{experiment_path}/J_{args.seed}.npy")
    else:
        first_epoch = 1
        last_epoch = p["n_epochs"] // 2

        sample_key, exploration_key = jax.random.split(key)
        n_gradient_steps = 0
        l2_losses = np.ones((p["n_epochs"], p["n_gradient_steps_per_epoch"])) * np.nan
        average_rewards = np.ones(p["n_epochs"]) * np.nan

        env.collect_random_samples(sample_key, replay_buffer, p["n_initial_samples"], p["horizon"])

    epsilon_schedule = EpsilonGreedySchedule(
        p["starting_eps"],
        p["ending_eps"],
        p["duration_eps"],
        exploration_key,
        n_gradient_steps,
    )

    for idx_epoch in tqdm(range(first_epoch, last_epoch + 1)):
        sum_reward = 0
        n_episodes = 0
        idx_training_step = 0
        absorbing = False

        while idx_training_step < p["n_gradient_steps_per_epoch"] or not absorbing:
            sample_key, key = jax.random.split(sample_key)
            reward, absorbing = env.collect_one_sample(q, q.params, p["horizon"], replay_buffer, epsilon_schedule)

            batch_samples = replay_buffer.sample_random_batch(key, p["batch_size"])
            q.params, q.optimizer_state, l2_loss = q.learn_on_batch(
                q.params, q.target_params, q.optimizer_state, batch_samples
            )

            n_gradient_steps += 1
            l2_losses[idx_epoch - 1, idx_training_step % p["n_gradient_steps_per_epoch"]] = l2_loss
            sum_reward += reward
            n_episodes += int(absorbing)
            idx_training_step += 1

            q.update_target_params(n_gradient_steps)

        average_rewards[idx_epoch] = sum_reward / n_episodes
        np.save(
            f"{experiment_path}/J_{args.seed}.npy",
            average_rewards,
        )
        np.save(
            f"{experiment_path}/L_{args.seed}.npy",
            l2_losses,
        )

    if not args.restart_training:
        np.save(f"{experiment_path}/K_{args.seed}", np.array([sample_key, epsilon_schedule.key]))
        env.save(f"{experiment_path}/E_{args.seed}")
        replay_buffer.save(f"{experiment_path}/R_{args.seed}")
        save_pickled_data(f"{experiment_path}/P_{args.seed}_{idx_epoch}", q.params)
        save_pickled_data(f"{experiment_path}/TP_{args.seed}_{idx_epoch}", q.target_params)
        save_pickled_data(f"{experiment_path}/O_{args.seed}", q.optimizer_state)
        np.save(f"{experiment_path}/N_{args.seed}", np.array(n_gradient_steps))
