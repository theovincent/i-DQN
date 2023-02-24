import sys
import argparse
import json
import jax
import jax.numpy as jnp
import numpy as np

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train iDQN on Atari.")
    addparse(parser)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, "iDQN", "Atari", args.bellman_iterations_scope, args.seed)
    p = json.load(
        open(f"experiments/atari/figures/{args.experiment_name.split('/')[0]}/parameters.json")
    )  # p for parameters

    from experiments.atari.utils import (
        define_environment,
        define_multi_q,
        collect_random_samples,
        collect_samples_multi_head,
        generate_keys,
    )
    from idqn.sample_collection.replay_buffer import ReplayBuffer
    from idqn.utils.importance_iteration import importance_bound
    from experiments.base.iDQN import train

    sample_key, exploration_key, q_key = generate_keys(args.seed)

    env = define_environment(jax.random.PRNGKey(p["env_seed"]), args.experiment_name.split("/")[1], p["gamma"])
    replay_buffer = ReplayBuffer(
        p["max_size"],
        f"experiments/atari/figures/{args.experiment_name}/iDQN/{args.bellman_iterations_scope}_R_{args.seed}_1",
        (env.n_stacked_frames, env.state_height, env.state_width),
        np.uint8,
        np.int8,
        overwrite=True,
    )
    collect_random_samples(env, sample_key, replay_buffer, p["n_initial_samples"], p["horizon"])

    if p["importance_iteration"] == "bound":
        importance_iteration = importance_bound(p["gamma"], args.bellman_iterations_scope)
    elif p["importance_iteration"] == "uniform":
        importance_iteration = jnp.ones(args.bellman_iterations_scope)
    q = define_multi_q(
        importance_iteration,
        (env.n_stacked_frames, env.state_height, env.state_width),
        env.n_actions,
        p["gamma"],
        q_key,
        p["n_shared_layers"],
        p["learning_rate"],
    )

    train("atari", args, q, p, exploration_key, sample_key, replay_buffer, collect_samples_multi_head, env)
