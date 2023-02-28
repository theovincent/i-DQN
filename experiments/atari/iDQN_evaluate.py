import os
import shutil
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

    parser = argparse.ArgumentParser("Evaluate iDQN on Atari.")
    addparse(parser)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, "iDQN", "Atari", args.bellman_iterations_scope, args.seed, train=False)
    p = json.load(
        open(f"experiments/atari/figures/{args.experiment_name.split('/')[0]}/parameters.json")
    )  # p for parameters

    from experiments.atari.utils import define_environment, define_multi_q
    from idqn.utils.params import load_params

    env = define_environment(jax.random.PRNGKey(p["env_seed"]), args.experiment_name.split("/")[1], 1)
    q = define_multi_q(
        jnp.zeros(args.bellman_iterations_scope),
        (env.n_stacked_frames, env.state_height, env.state_width),
        env.n_actions,
        p["gamma"],
        jax.random.PRNGKey(0),
        p["n_shared_layers"],
    )

    js = np.nan * np.zeros(p["n_epochs"] + 1)
    max_j = -float("inf")
    argmax_j = None
    path_params = (
        f"experiments/atari/figures/{args.experiment_name}/iDQN/{args.bellman_iterations_scope}_P_{args.seed}_"
    )
    if args.restart_training:
        idx_epoch = p["n_epochs"] // 2 + 1
        last_epoch = p["n_epochs"]
    else:
        idx_epoch = 0
        last_epoch = p["n_epochs"] // 2
    list_idx_epoch_video = np.ceil(np.linspace(idx_epoch, last_epoch, 5))

    while idx_epoch <= last_epoch:
        if os.path.exists(path_params + str(idx_epoch)):
            params = load_params(path_params + str(idx_epoch))
            js[idx_epoch] = env.evaluate(
                q,
                0,
                params,
                p["horizon"],
                p["n_simulations"],
                f"{args.experiment_name}/iDQN/K{args.bellman_iterations_scope}_{idx_epoch}_s{args.seed}"
                if idx_epoch in list_idx_epoch_video
                else None,
            )

            if max_j < js[idx_epoch]:
                if argmax_j is not None:
                    os.remove(f"{path_params}{argmax_j}_best")

                argmax_j = idx_epoch
                max_j = js[idx_epoch]
                shutil.copy(path_params + str(idx_epoch), f"{path_params}{argmax_j}_best")

            np.save(
                f"experiments/atari/figures/{args.experiment_name}/iDQN/{args.bellman_iterations_scope}_J_{args.seed}.npy",
                js,
            )
            os.remove(path_params + str(idx_epoch))
            print(f"Epoch {idx_epoch} done")
            idx_epoch += 1
        else:
            print("Sleep")
            time.sleep(60)

    if args.restart_training:
        os.remove(path_params.replace("_P_", "_R_") + "*")
        os.remove(path_params.replace("_P_", "_K_") + ".npy")
        os.remove(path_params.replace("_P_", "_O_"))
        os.remove(path_params.replace("_P_", "_E_"))
