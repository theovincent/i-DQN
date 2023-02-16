import sys
import argparse
import json
import jax
import jax.numpy as jnp

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train iDQN on Lunar Lander.")
    addparse(parser)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, "iDQN", "Lunar Lander", args.bellman_iterations_scope, args.seed)
    p = json.load(open(f"experiments/lunar_lander/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.lunar_lander.utils import (
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

    env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["gamma"])
    replay_buffer = ReplayBuffer(p["max_size"])
    collect_random_samples(env, sample_key, replay_buffer, p["n_initial_samples"], p["horizon"])

    if p["importance_iteration"] == "bound":
        importance_iteration = importance_bound(p["gamma"], args.bellman_iterations_scope)
    elif p["importance_iteration"] == "uniform":
        importance_iteration = jnp.ones(args.bellman_iterations_scope)
    q = define_multi_q(
        importance_iteration,
        p["gamma"],
        q_key,
        p["shared_layers_dimension"],
        p["layers_dimension"],
        learning_rate=p["learning_rate"],
    )

    train("lunar_lander", args, q, p, exploration_key, sample_key, replay_buffer, collect_samples_multi_head, env)
