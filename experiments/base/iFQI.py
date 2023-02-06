from tqdm import tqdm
import numpy as np
from argparse import Namespace

from idqn.sample_collection.dataloader import SampleDataLoader
from idqn.networks.base_q import iQ
from idqn.utils.params import save_params


def train(environment_name: str, args: Namespace, q: iQ, p: dict, data_loader_samples: SampleDataLoader) -> None:
    n_training_steps = p["forward_moves"] * args.bellman_iterations_scope * p["fitting_steps"]
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

            n_steps += 1

            # Target update
            if n_steps % p["target_update_frequency"] == 0:
                params_target = q.params

            # Move forward in the number of Bellman iterations
            if n_steps % (args.bellman_iterations_scope * p["fitting_steps"]) == 0:
                save_params(
                    f"experiments/{environment_name}/figures/{args.experiment_name}/IFQI/{args.max_bellman_iterations}_P_{args.seed}_{n_steps}",
                    q.params,
                )
                q.params = q.move_forward(q.params)
                params_target = q.params

            l2_losses[n_steps] = cumulative_l2_loss

    np.save(
        f"experiments/{environment_name}/figures/{args.experiment_name}/IFQI/{args.max_bellman_iterations}_L_{args.seed}.npy",
        l2_losses,
    )
