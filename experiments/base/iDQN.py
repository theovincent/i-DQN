from tqdm import tqdm
import numpy as np
import jax
import optax
from argparse import Namespace

from pbo.sample_collection.exploration import EpsilonGreedySchedule
from pbo.sample_collection.replay_buffer import ReplayBuffer
from pbo.networks.base_q import BaseQ
from pbo.utils.params import save_params


def train(
    environment_name: str,
    args: Namespace,
    q: BaseQ,
    p: dict,
    exploration_key: jax.random.PRNGKeyArray,
    sample_key: jax.random.PRNGKeyArray,
    replay_buffer: ReplayBuffer,
    collect_samples_multi_head,
    env,
) -> None:
    epsilon_schedule = EpsilonGreedySchedule(
        p["starting_eps_idqn"],
        p["ending_eps_idqn"],
        p["training_steps_idqn"] * p["fitting_steps_idqn"] * p["steps_per_update_idqn"],
        exploration_key,
    )

    l2_losses = np.ones((p["training_steps_idqn"], p["fitting_steps_idqn"])) * np.nan

    for training_step in tqdm(range(p["training_steps_idqn"])):
        params_target = q.params

        for fitting_step in tqdm(range(p["fitting_steps_idqn"]), leave=False):
            sample_key, key = jax.random.split(sample_key)
            if jax.random.uniform(key) <= p["steps_per_update_idqn"]:
                collect_samples_multi_head(
                    env,
                    replay_buffer,
                    q,
                    q.params,
                    1 if p["steps_per_update_idqn"] < 1 else p["steps_per_update_idqn"],
                    p["horizon"],
                    epsilon_schedule,
                )

            sample_key, key = jax.random.split(sample_key)
            batch_samples = replay_buffer.sample_random_batch(key, p["batch_size_samples"])

            q.params, q.optimizer_state, l2_loss = q.learn_on_batch(
                q.params, params_target, q.optimizer_state, batch_samples
            )

            l2_losses[training_step, fitting_step] = l2_loss

    save_params(
        f"experiments/{environment_name}/figures/{args.experiment_name}/IDQN/{args.max_bellman_iterations}_P_{args.seed}",
        q.params,
    )
    np.save(
        f"experiments/{environment_name}/figures/{args.experiment_name}/IDQN/{args.max_bellman_iterations}_L_{args.seed}.npy",
        l2_losses,
    )
