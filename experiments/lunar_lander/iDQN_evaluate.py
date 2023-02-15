import os
import sys
import argparse
import time
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

    js = np.nan * np.zeros(p["n_epochs"] + 1)
    path_params = (
        f"experiments/lunar_lander/figures/{args.experiment_name}/iDQN/{args.bellman_iterations_scope}_P_{args.seed}_"
    )
    idx_epoch = 0

    while idx_epoch <= p["n_epochs"]:
        if os.path.exists(path_params + str(idx_epoch)):
            params = load_params(path_params + str(idx_epoch))
            js[idx_epoch] = env.evaluate(
                q,
                0,
                params,
                p["horizon"],
                p["n_simulations"],
                video_path=f"{args.experiment_name}/iDQN/K{args.bellman_iterations_scope}_{idx_epoch}_{args.seed}",
            )

            np.save(
                f"experiments/lunar_lander/figures/{args.experiment_name}/iDQN/{args.bellman_iterations_scope}_J_{args.seed}.npy",
                js,
            )
            idx_epoch += 1
        else:
            time.sleep(60)
