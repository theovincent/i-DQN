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
    experiment_path = (
        f"experiments/{environment_name}/figures/{args.experiment_name}/DQN/{args.bellman_iterations_scope}"
    )

    if args.restart_training:
        first_epoch = p["n_epochs"] // 2 + 1
        last_epoch = p["n_epochs"]

        env.reset_key, sample_key, exploration_key = np.load(f"{experiment_path}_K_{args.seed}.npy")
        q.params = load_pickled_data(f"{experiment_path}_P_{args.seed}_{first_epoch - 1}")
        q.optimizer_state = load_pickled_data(f"{experiment_path}_O_{args.seed}")
        l2_losses = np.load(f"{experiment_path}_L_{args.seed}.npy")
        env.load(f"{experiment_path}_E_{args.seed}")

        replay_buffer.load(
            f"experiments/atari/figures/{args.experiment_name}/DQN/{args.bellman_iterations_scope}_R_{args.seed}"
        )
    else:
        first_epoch = 1
        last_epoch = p["n_epochs"] // 2

        sample_key, exploration_key = jax.random.split(key)
        save_pickled_data(f"{experiment_path}_P_{args.seed}_{first_epoch - 1}", q.params)
        l2_losses = np.ones((p["n_epochs"], p["n_gradient_steps_per_epoch"])) * np.nan

        env.collect_random_samples(sample_key, replay_buffer, p["n_initial_samples"], p["horizon"])

    params_target = q.params
    n_gradient_steps = (first_epoch - 1) * p["n_gradient_steps_per_epoch"]
    epsilon_schedule = EpsilonGreedySchedule(
        p["starting_eps"],
        p["ending_eps"],
        p["duration_eps"],
        exploration_key,
        n_gradient_steps * p["n_samples_per_gradient_step"],
    )

    for idx_epoch in tqdm(range(first_epoch, last_epoch + 1)):
        for idx_training_step in tqdm(range(p["n_gradient_steps_per_epoch"]), leave=False):
            sample_key, key = jax.random.split(sample_key)
            env.collect_samples(
                q, q.params, p["n_samples_per_gradient_step"], p["horizon"], replay_buffer, epsilon_schedule
            )
            batch_samples = replay_buffer.sample_random_batch(key, p["batch_size"])

            q.params, q.optimizer_state, l2_loss = q.learn_on_batch(
                q.params, params_target, q.optimizer_state, batch_samples
            )

            l2_losses[idx_epoch - 1, idx_training_step] = l2_loss
            n_gradient_steps += 1

            if n_gradient_steps % p["dqn_n_gradient_steps_per_target_update"] == 0:
                params_target = q.params

        save_pickled_data(
            f"{experiment_path}_P_{args.seed}_{idx_epoch}",
            q.params,
        )
        np.save(
            f"{experiment_path}_L_{args.seed}.npy",
            l2_losses,
        )

    replay_buffer.save(
        f"experiments/atari/figures/{args.experiment_name}/DQN/{args.bellman_iterations_scope}_R_{args.seed}"
    )
    np.save(f"{experiment_path}_K_{args.seed}", np.array([env.reset_key, sample_key, epsilon_schedule.key]))
    save_pickled_data(f"{experiment_path}_O_{args.seed}", q.optimizer_state)
    env.save(f"{experiment_path}_E_{args.seed}")
