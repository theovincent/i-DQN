from tqdm import tqdm
import numpy as np
from argparse import Namespace

from idqn.sample_collection.dataloader import SampleDataLoader
from idqn.networks.base_q import iQ
from idqn.utils.params import save_params


def train(environment_name: str, args: Namespace, q: iQ, p: dict, data_loader_samples: SampleDataLoader) -> None:
    assert (
        p["n_bellman_iterations_per_epoch"] % args.bellman_iterations_scope == 0
    ), f"n_bellman_iterations_per_epoch: {p['n_bellman_iterations_per_epoch']} shoud be a multiple of bellman_iterations_scope: {args.bellman_iterations_scope}."

    n_training_steps = p["n_bellman_iterations_per_epoch"] * p["fitting_steps_per_bellman_iteration"]
    l2_losses = np.ones(p["n_epochs"] * n_training_steps) * np.nan
    n_steps = 0
    params_target = q.params

    for _ in tqdm(range(p["n_epochs"])):
        for _ in tqdm(range(n_training_steps), leave=False):
            cumulative_l2_loss = 0

            data_loader_samples.shuffle()
            for batch_samples in data_loader_samples:
                q.params, q.optimizer_state, l2_loss = q.learn_on_batch(
                    q.params, params_target, q.optimizer_state, batch_samples
                )
                cumulative_l2_loss += l2_loss

            l2_losses[n_steps] = cumulative_l2_loss
            n_steps += 1

            # Target update
            if n_steps % p["target_update_frequency"] == 0:
                params_target = q.params

            # Move forward in the number of Bellman iterations
            if n_steps % (args.bellman_iterations_scope * p["fitting_steps_per_bellman_iteration"]) == 0:
                n_forward_moves = (
                    n_steps // (args.bellman_iterations_scope * p["fitting_steps_per_bellman_iteration"]) - 1
                )
                start_k = n_forward_moves * args.bellman_iterations_scope
                end_k = start_k + args.bellman_iterations_scope
                save_params(
                    f"experiments/{environment_name}/figures/{args.experiment_name}/iFQI/{args.bellman_iterations_scope}_P_{args.seed}_{start_k}-{end_k}",
                    q.params,
                )
                q.params = q.move_forward(q.params)
                params_target = q.params

    np.save(
        f"experiments/{environment_name}/figures/{args.experiment_name}/iFQI/{args.bellman_iterations_scope}_L_{args.seed}.npy",
        l2_losses,
    )
