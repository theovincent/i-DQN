import os
from tqdm import tqdm
import numpy as np
import jax
from argparse import Namespace

from idqn.sample_collection.exploration import EpsilonGreedySchedule
from idqn.sample_collection.replay_buffer import ReplayBuffer
from idqn.networks.base import BaseQ
from idqn.environments.atari import AtariEnv


def train(
    key: jax.random.PRNGKey,
    experiment_path: str,
    args: Namespace,
    p: dict,
    q: BaseQ,
    env: AtariEnv,
    replay_buffer: ReplayBuffer,
) -> None:
    sample_key, exploration_key = jax.random.split(key)
    n_training_steps = 0
    losses = np.zeros((p["n_epochs"], p["n_training_steps_per_epoch"])) * np.nan
    js = np.zeros(p["n_epochs"]) * np.nan
    stds = np.zeros(p["n_epochs"]) * np.nan
    approximation_errors = np.zeros(p["n_epochs"]) * np.nan
    max_j = -float("inf")
    argmax_j = None

    env.collect_random_samples(sample_key, replay_buffer, p["n_initial_samples"], p["horizon"])

    epsilon_schedule = EpsilonGreedySchedule(
        p["starting_eps"],
        p["ending_eps"],
        p["duration_eps"],
        exploration_key,
        n_training_steps,
    )

    for idx_epoch in tqdm(range(p["n_epochs"])):
        sum_reward = 0
        n_episodes = 0
        idx_training_step = 0
        has_reset = False

        while idx_training_step < p["n_training_steps_per_epoch"] or not has_reset:
            reward, has_reset = env.collect_one_sample(q, q.params, p["horizon"], replay_buffer, epsilon_schedule)

            sum_reward += reward
            n_episodes += int(has_reset)

            losses[
                idx_epoch, np.minimum(idx_training_step, p["n_training_steps_per_epoch"] - 1)
            ] = q.update_online_params(n_training_steps, replay_buffer, has_reset=has_reset)
            q.update_target_params(n_training_steps)

            idx_training_step += 1
            n_training_steps += 1

        js[idx_epoch] = sum_reward / n_episodes
        np.save(
            f"{experiment_path}J_{args.seed}.npy",
            js,
        )
        np.save(
            f"{experiment_path}L_{args.seed}.npy",
            losses,
        )
        if js[idx_epoch] > max_j:
            if argmax_j is not None:
                os.remove(f"{experiment_path}Q_{args.seed}_{argmax_j}_best_online_params")

            argmax_j = idx_epoch
            max_j = js[idx_epoch]
            q.save(f"{experiment_path}Q_{args.seed}_{argmax_j}_best")

        if args.bellman_iterations_scope is not None and p.get("compute_head_std", False):
            stds[idx_epoch] = q.compute_standard_deviation_head(replay_buffer, key)
            np.save(
                f"{experiment_path}S_{args.seed}.npy",
                stds,
            )
        if args.bellman_iterations_scope is not None and p.get("compute_approximation_error", False):
            approximation_errors[idx_epoch] = q.compute_approximation_error(replay_buffer, key)
            np.save(
                f"{experiment_path}A_{args.seed}.npy",
                approximation_errors,
            )
