import sys
import argparse
import multiprocessing
import json
import jax
import jax.numpy as jnp
import numpy as np

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Evaluate iDQN on Lunar Lander.")
    addparse(parser)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, "iDQN", "Lunar Lander", args.bellman_iterations_scope, args.seed, train=False)
    p = json.load(open(f"experiments/lunar_lander/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.lunar_lander.utils import define_environment, define_multi_q
    from idqn.utils.params import load_params

    env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["gamma"])
    q = define_multi_q(
        jnp.zeros(args.bellman_iterations_scope),
        p["gamma"],
        jax.random.PRNGKey(0),
        p["shared_layers_dimension"],
        p["layers_dimension"],
    )

    def evaluate(
        iteration: int,
        idx_head: int,
        j_list: list,
        path_params: str,
    ):
        params = load_params(path_params)
        j_list[iteration] = env.evaluate(
            q,
            idx_head,
            params,
            p["horizon"],
            p["n_simulations"],
            video_path=f"{args.experiment_name}/iDQN/K{args.bellman_iterations_scope}_{iteration}_{args.seed}",
        )

        print(f"Iteration {iteration} done")

    manager = multiprocessing.Manager()
    iterated_j = manager.list(list(np.nan * np.zeros(p["n_epochs"] * p["n_bellman_iterations_per_epoch"] + 1)))

    processes = []
    n_forward_moves = p["n_epochs"] * p["n_bellman_iterations_per_epoch"] // args.bellman_iterations_scope

    path_params = f"experiments/lunar_lander/figures/{args.experiment_name}/iDQN/{args.bellman_iterations_scope}_P_{args.seed}_0-{args.bellman_iterations_scope}"
    processes.append(
        multiprocessing.Process(
            target=evaluate,
            args=(0, 0, iterated_j, path_params),
        )
    )

    for idx_forward_move in range(n_forward_moves):
        start_k = idx_forward_move * args.bellman_iterations_scope
        end_k = start_k + args.bellman_iterations_scope
        path_params = f"experiments/lunar_lander/figures/{args.experiment_name}/iDQN/{args.bellman_iterations_scope}_P_{args.seed}_{start_k}-{end_k}"

        for idx_head in range(1, args.bellman_iterations_scope + 1):
            iteration = idx_head + idx_forward_move * args.bellman_iterations_scope
            processes.append(
                multiprocessing.Process(
                    target=evaluate,
                    args=(iteration, idx_head, iterated_j, path_params),
                )
            )

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    np.save(
        f"experiments/lunar_lander/figures/{args.experiment_name}/iDQN/{args.bellman_iterations_scope}_J_{args.seed}.npy",
        iterated_j,
    )
