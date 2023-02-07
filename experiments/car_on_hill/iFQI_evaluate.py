import sys
import argparse
import multiprocessing
import json
import jax
import haiku as hk
import numpy as np

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    with jax.default_device(jax.devices("cpu")[0]):
        import warnings

        warnings.simplefilter(action="ignore", category=FutureWarning)

        parser = argparse.ArgumentParser("Evaluate IFQI on Car-On-Hill.")
        addparse(parser)
        args = parser.parse_args(argvs)
        print_info(args.experiment_name, "IFQI", "Car-On-Hill", args.bellman_iterations_scope, args.seed, train=False)
        p = json.load(
            open(f"experiments/car_on_hill/figures/{args.experiment_name}/parameters.json")
        )  # p for parameters

        from experiments.car_on_hill.utils import define_environment, define_multi_q
        from idqn.networks.learnable_multi_head_q import FullyConnectedMultiQ
        from idqn.utils.params import load_params

        env, states_x, _, states_v, _ = define_environment(p["gamma"], p["n_states_x"], p["n_states_v"])

        q = define_multi_q(
            args.bellman_iterations_scope + 1,
            p["gamma"],
            jax.random.PRNGKey(0),
            p["layers_dimension"],
        )

        def evaluate(
            iteration: int,
            idx_head: int,
            v_list: list,
            q_estimate_list: list,
            q: FullyConnectedMultiQ,
            params: hk.Params,
            horizon: int,
            states_x: np.ndarray,
            states_v: np.ndarray,
        ):
            v_list[iteration] = env.v_mesh_multi_head(q, idx_head, params, horizon, states_x, states_v)
            q_estimate_list[iteration] = env.q_multi_head_estimate_mesh(q, idx_head, params, states_x, states_v)

        manager = multiprocessing.Manager()
        iterated_v = manager.list(
            list(
                np.nan
                * np.zeros((p["n_epochs"] * p["n_bellman_iterations_per_epoch"] + 1, p["n_states_x"], p["n_states_v"]))
            )
        )
        iterated_q_estimate = manager.list(
            list(
                np.nan
                * np.zeros(
                    (
                        p["n_epochs"] * p["n_bellman_iterations_per_epoch"] + 1,
                        p["n_states_x"],
                        p["n_states_v"],
                        env.n_actions,
                    )
                )
            )
        )

        processes = []
        n_forward_moves = p["n_epochs"] * p["n_bellman_iterations_per_epoch"] // args.bellman_iterations_scope

        params = load_params(
            f"experiments/car_on_hill/figures/{args.experiment_name}/iFQI/{args.bellman_iterations_scope}_P_{args.seed}_{0}-{args.bellman_iterations_scope}"
        )
        processes.append(
            multiprocessing.Process(
                target=evaluate,
                args=(0, 0, iterated_v, iterated_q_estimate, q, params, p["horizon"], states_x, states_v),
            )
        )

        for idx_forward_move in range(n_forward_moves):
            start_k = idx_forward_move * args.bellman_iterations_scope
            end_k = start_k + args.bellman_iterations_scope
            params = load_params(
                f"experiments/car_on_hill/figures/{args.experiment_name}/iFQI/{args.bellman_iterations_scope}_P_{args.seed}_{start_k}-{end_k}"
            )
            for idx_head in range(1, args.bellman_iterations_scope + 1):
                iteration = idx_head + idx_forward_move * args.bellman_iterations_scope
                processes.append(
                    multiprocessing.Process(
                        target=evaluate,
                        args=(
                            iteration,
                            idx_head,
                            iterated_v,
                            iterated_q_estimate,
                            q,
                            params,
                            p["horizon"],
                            states_x,
                            states_v,
                        ),
                    )
                )

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        np.save(
            f"experiments/car_on_hill/figures/{args.experiment_name}/iFQI/{args.bellman_iterations_scope}_V_{args.seed}.npy",
            iterated_v,
        )
        np.save(
            f"experiments/car_on_hill/figures/{args.experiment_name}/iFQI/{args.bellman_iterations_scope}_Q_{args.seed}.npy",
            iterated_q_estimate,
        )
