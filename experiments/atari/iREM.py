import sys
import argparse
import json
import numpy as np

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train iREM on Atari.")
    addparse(parser)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, "iREM", "Atari", args.bellman_iterations_scope, args.seed)
    p = json.load(
        open(f"experiments/atari/figures/{args.experiment_name.split('/')[0]}/parameters.json")
    )  # p for parameters

    from experiments.atari.utils import generate_keys
    from idqn.environments.atari import AtariEnv
    from idqn.sample_collection.replay_buffer import ReplayBuffer
    from idqn.networks.architectures.irem import AtariiREM
    from idqn.utils.head_behaviorial_policy import head_behaviorial_policy
    from experiments.base.DQN import train

    q_key, train_key = generate_keys(args.seed)

    env = AtariEnv(args.experiment_name.split("/")[1])

    replay_buffer = ReplayBuffer(
        (env.state_height, env.state_width),
        p["replay_buffer_size"],
        p["batch_size"],
        p["irem_n_step_return"],
        p["gamma"],
        lambda x: np.clip(x, -1, 1),
    )

    q = AtariiREM(
        args.bellman_iterations_scope + 1,
        (env.state_height, env.state_width, env.n_stacked_frames),
        env.n_actions,
        p["gamma"],
        q_key,
        head_behaviorial_policy(p["irem_head_behaviorial_policy"], args.bellman_iterations_scope + 1),
        p["irem_learning_rate"],
        p["irem_optimizer_eps"],
        p["n_training_steps_per_online_update"],
        p["irem_n_training_steps_per_target_update"],
        p["irem_n_training_steps_per_rolling_step"],
        p["irem_shared_network"],
    )

    train(
        train_key,
        f"experiments/atari/figures/{args.experiment_name}/iREM/{args.bellman_iterations_scope}_",
        args,
        p,
        q,
        env,
        replay_buffer,
    )
