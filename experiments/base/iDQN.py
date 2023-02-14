from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp
from argparse import Namespace

from idqn.sample_collection.exploration import EpsilonGreedySchedule
from idqn.utils.importance_iteration import importance_bound
from idqn.sample_collection.replay_buffer import ReplayBuffer
from idqn.networks.base_q import iQ
from idqn.utils.params import save_params


def train(
    environment_name: str,
    args: Namespace,
    q: iQ,
    p: dict,
    exploration_key: jax.random.PRNGKeyArray,
    sample_key: jax.random.PRNGKeyArray,
    replay_buffer: ReplayBuffer,
    collect_samples_multi_head,
    env,
) -> None:
    assert (
        p["n_bellman_iterations_per_epoch"] % args.bellman_iterations_scope == 0
    ), f"n_bellman_iterations_per_epoch: {p['n_bellman_iterations_per_epoch']} shoud be a multiple of bellman_iterations_scope: {args.bellman_iterations_scope}."

    epsilon_schedule = EpsilonGreedySchedule(p["starting_eps"], p["ending_eps"], p["duration_eps"], exploration_key)

    if p["head_behaviorial_policy"] == "uniform":
        head_behaviorial_probability = jnp.ones(args.bellman_iterations_scope + 1)
    elif p["head_behaviorial_policy"] == "last":
        head_behaviorial_probability = jnp.zeros(args.bellman_iterations_scope + 1)
        head_behaviorial_probability = head_behaviorial_probability.at[args.bellman_iterations_scope].set(1)
    elif p["head_behaviorial_policy"] == "bound":
        head_behaviorial_probability = jnp.zeros(args.bellman_iterations_scope + 1)
        head_behaviorial_probability = head_behaviorial_probability.at[1 : args.bellman_iterations_scope].set(
            importance_bound(p["gamma"], args.bellman_iterations_scope)
        )

    n_training_steps = p["n_bellman_iterations_per_epoch"] * p["gradient_steps_per_bellman_iteration"]
    l2_losses = np.ones(p["n_epochs"] * n_training_steps) * np.nan
    n_gradient_steps = 0
    params_target = q.params

    for _ in tqdm(range(p["n_epochs"])):
        for _ in tqdm(range(n_training_steps), leave=False):
            sample_key, key = jax.random.split(sample_key)
            collect_samples_multi_head(
                env,
                replay_buffer,
                q,
                q.params,
                p["sampling_steps_per_update"],
                p["horizon"],
                epsilon_schedule,
                head_behaviorial_probability,
            )
            batch_samples = replay_buffer.sample_random_batch(key, p["batch_size"])

            q.params, q.optimizer_state, l2_loss = q.learn_on_batch(
                q.params, params_target, q.optimizer_state, batch_samples
            )

            l2_losses[n_gradient_steps] = l2_loss
            n_gradient_steps += 1

            if n_gradient_steps % p["target_update_per_gradient_step"] == 0:
                params_target = q.params
            if n_gradient_steps % (args.bellman_iterations_scope * p["gradient_steps_per_bellman_iteration"]) == 0:
                n_forward_moves = (
                    n_gradient_steps // (args.bellman_iterations_scope * p["gradient_steps_per_bellman_iteration"]) - 1
                )
                start_k = n_forward_moves * args.bellman_iterations_scope
                end_k = start_k + args.bellman_iterations_scope
                save_params(
                    f"experiments/{environment_name}/figures/{args.experiment_name}/iDQN/{args.bellman_iterations_scope}_P_{args.seed}_{start_k}-{end_k}",
                    q.params,
                )
                q.params = q.move_forward(q.params)
                params_target = q.params

    np.save(
        f"experiments/{environment_name}/figures/{args.experiment_name}/iDQN/{args.max_bellman_iterations}_L_{args.seed}.npy",
        l2_losses,
    )
