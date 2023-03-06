from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp
from argparse import Namespace

from idqn.sample_collection.exploration import EpsilonGreedySchedule
from idqn.utils.importance_iteration import importance_bound
from idqn.sample_collection.replay_buffer import ReplayBuffer
from idqn.networks.base_q import iQ
from idqn.utils.params import load_params, save_params
from idqn.utils.pickle import load_pickled_data, save_pickled_data


def train(
    environment_name: str,
    args: Namespace,
    q: iQ,
    p: dict,
    exploration_key: jax.random.PRNGKeyArray,
    sample_key: jax.random.PRNGKeyArray,
    replay_buffer: ReplayBuffer,
    collect_random_samples,
    collect_samples_multi_head,
    env,
) -> None:
    experiment_path = (
        f"experiments/{environment_name}/figures/{args.experiment_name}/iDQN/{args.bellman_iterations_scope}"
    )

    if args.restart_training:
        first_epoch = p["n_epochs"] // 2 + 1
        last_epoch = p["n_epochs"]

        env.reset_key, exploration_key, sample_key = np.load(f"{experiment_path}_K_{args.seed}.npy")
        q.params = load_params(f"{experiment_path}_P_{args.seed}_{first_epoch - 1}")
        q.optimizer_state = load_pickled_data(f"{experiment_path}_O_{args.seed}")
        l2_losses = np.load(f"{experiment_path}_L_{args.seed}.npy")
        env.load(f"{experiment_path}_E_{args.seed}")
    else:
        first_epoch = 1
        last_epoch = p["n_epochs"] // 2

        save_params(f"{experiment_path}_P_{args.seed}_{first_epoch - 1}", q.params)
        l2_losses = np.ones((p["n_epochs"], p["n_gradient_steps_per_epoch"])) * np.nan

        collect_random_samples(env, sample_key, replay_buffer, p["n_initial_samples"], p["horizon"])

    params_target = q.params
    n_gradient_steps = (first_epoch - 1) * p["n_gradient_steps_per_epoch"]
    epsilon_schedule = EpsilonGreedySchedule(
        p["starting_eps"],
        p["ending_eps"],
        p["duration_eps"],
        exploration_key,
        n_gradient_steps * p["n_samples_per_gradient_step"],
    )

    if p["head_behaviorial_policy"] == "uniform":
        head_behaviorial_probability = jnp.ones(args.bellman_iterations_scope + 1)
    elif p["head_behaviorial_policy"] == "last":
        head_behaviorial_probability = jnp.zeros(args.bellman_iterations_scope + 1)
        head_behaviorial_probability = head_behaviorial_probability.at[args.bellman_iterations_scope].set(1)
    elif p["head_behaviorial_policy"] == "bound":
        head_behaviorial_probability = jnp.zeros(args.bellman_iterations_scope + 1)
        head_behaviorial_probability = head_behaviorial_probability.at[1:].set(
            importance_bound(p["gamma"], args.bellman_iterations_scope)
        )

    for idx_epoch in tqdm(range(first_epoch, last_epoch + 1)):
        for idx_training_step in tqdm(range(p["n_gradient_steps_per_epoch"]), leave=False):
            sample_key, key = jax.random.split(sample_key)
            collect_samples_multi_head(
                env,
                replay_buffer,
                q,
                q.params,
                p["n_samples_per_gradient_step"],
                p["horizon"],
                epsilon_schedule,
                head_behaviorial_probability,
            )
            batch_samples = replay_buffer.sample_random_batch(key, p["batch_size"])

            q.params, q.optimizer_state, l2_loss = q.learn_on_batch(
                q.params, params_target, q.optimizer_state, batch_samples
            )

            l2_losses[idx_epoch - 1, idx_training_step] = l2_loss
            n_gradient_steps += 1

            if n_gradient_steps % p["n_gradient_steps_per_target_update"] == 0:
                params_target = q.params
            if (
                n_gradient_steps
                % (jnp.ceil(p["n_gradient_steps_per_head_update"] * args.bellman_iterations_scope / p["speed"]))
                == 0
            ):
                q.params = q.move_forward(q.params)
                params_target = q.params

        save_params(
            f"{experiment_path}_P_{args.seed}_{idx_epoch}",
            q.params,
        )

        if idx_epoch == last_epoch:
            replay_buffer.save()
            np.save(f"{experiment_path}_K_{args.seed}", np.array([env.reset_key, epsilon_schedule.key, sample_key]))
            save_pickled_data(f"{experiment_path}_O_{args.seed}", q.optimizer_state)
            env.save(f"{experiment_path}_E_{args.seed}")

        np.save(
            f"{experiment_path}_L_{args.seed}.npy",
            l2_losses,
        )
