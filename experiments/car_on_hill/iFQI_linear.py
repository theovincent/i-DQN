import sys
import argparse
import json
import math
import jax

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train iFQI linear on Car-On-Hill.")
    addparse(parser)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, "iFQI linear", "Car-On-Hill", args.bellman_iterations_scope, args.seed)
    p = json.load(
        open(f"experiments/car_on_hill/figures/{args.experiment_name.split('/')[0]}/parameters.json")
    )  # p for parameters

    from idqn.environments.car_on_hill import CarOnHillEnv
    from idqn.sample_collection.replay_buffer import ReplayBuffer
    from idqn.networks.architectures.ifqi import CarOnHilliFQI
    from experiments.base.FQI_linear import train

    q_key = jax.random.PRNGKey(args.seed)

    env = CarOnHillEnv(p["gamma"])

    replay_buffer = ReplayBuffer(
        (2,),
        p["replay_buffer_size"],
        p["batch_size"],
        p["n_step_return"],
        p["gamma"],
        lambda x: x,
        stack_size=1,
        observation_dtype=float,
    )

    q = CarOnHilliFQI(
        args.bellman_iterations_scope + 1,
        p["features_path"],
        (2,),
        env.n_actions,
        math.pow(p["gamma"], p["n_step_return"]),
        q_key,
        p["learning_rate"],
        p["optimizer_eps"],
    )

    train(f"experiments/car_on_hill/figures/{args.experiment_name}/iFQI_linear/", args, p, q, env, replay_buffer)
