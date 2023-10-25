import sys
import argparse
import json
import numpy as np

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train IQN on Atari.")
    addparse(parser)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, "IQN", "Atari", args.bellman_iterations_scope, args.seed)
    p = json.load(
        open(f"experiments/atari/figures/{args.experiment_name.split('/')[0]}/parameters.json")
    )  # p for parameters

    from experiments.atari.utils import generate_keys
    from idqn.environments.atari import AtariEnv
    from idqn.sample_collection.replay_buffer import ReplayBuffer
    from idqn.networks.architectures.iqn import AtariIQN
    from experiments.base.DQN import train

    q_key, train_key = generate_keys(args.seed)

    env = AtariEnv(args.experiment_name.split("/")[1])

    replay_buffer = ReplayBuffer(
        (env.state_height, env.state_width),
        p["replay_buffer_size"],
        p["batch_size"],
        p["iqn_n_step_return"],
        p["gamma"],
        lambda x: np.clip(x, -1, 1),
    )

    q = AtariIQN(
        (env.state_height, env.state_width, env.n_stacked_frames),
        env.n_actions,
        p["gamma"],
        q_key,
        p["iqn_learning_rate"],
        p["iqn_optimizer_eps"],
        p["n_training_steps_per_online_update"],
        p["iqn_n_training_steps_per_target_update"],
    )

    train(train_key, f"experiments/atari/figures/{args.experiment_name}/IQN/", args, p, q, env, replay_buffer)
