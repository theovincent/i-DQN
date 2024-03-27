from typing import Tuple
import sys
import argparse
import json
import math
import numpy as np
import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from experiments.base.parser import addparse
from experiments.base.print import print_info
from idqn.sample_collection import IDX_RB
from idqn.networks.architectures.ifqi import CarOnHilliFQI


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train iFQI on Car-On-Hill.")
    addparse(parser)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, "iFQI", "Car-On-Hill", args.bellman_iterations_scope, args.seed)
    p = json.load(
        open(f"experiments/car_on_hill/figures/{args.experiment_name.split('/')[0]}/parameters.json")
    )  # p for parameters

    from idqn.environments.car_on_hill import CarOnHillEnv
    from idqn.sample_collection.replay_buffer import ReplayBuffer

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
        2,
        p["features_path"],
        (2,),
        env.n_actions,
        math.pow(p["gamma"], p["n_step_return"]),
        q_key,
        p["learning_rate"],
        p["optimizer_eps"],
    )

    env.collect_random_samples(
        jax.random.PRNGKey(0),
        replay_buffer,
        p["n_random_samples"],
        p["n_oriented_samples"],
        p["oriented_states"],
        p["horizon"],
    )

    save_path = f"experiments/car_on_hill/figures/{args.experiment_name}/iFQI_linear/optimal_Q_r"
    q.save(f"{save_path}0")
    dataset = replay_buffer.get_all_valid_samples()

    for bellman_iteration in range(1, p["n_bellman_iterations"] + 1):
        update_bellman_iteration(q, dataset)

        q.save(f"{save_path}{bellman_iteration}")


def update_bellman_iteration(q: CarOnHilliFQI, params: FrozenDict, dataset: Tuple):
    # shape (n_samples, 1)
    targets = jax.vmap(q.compute_target, in_axes=(None, 0))(params, dataset)
    # shape (n_features, n_samples)
    features = jax.vmap(
        lambda sample: q.network.feature_net.apply(q.network.feature_net.params, jnp.squeeze(sample[IDX_RB["state"]])),
        out_axes=1,
    )(dataset)

    idx_action_0 = dataset[IDX_RB["action"]] == 0
    targets_action_0 = targets[idx_action_0]
    targets_action_1 = targets[~idx_action_0]
    features_action_0 = features[:, idx_action_0]
    features_action_1 = features[:, ~idx_action_0]

    # shape (n_features)
    params_action_0 = np.linalg.inv(features_action_0 @ features_action_0.T) @ features_action_0 @ targets_action_0
    params_action_1 = np.linalg.inv(features_action_1 @ features_action_1.T) @ features_action_1 @ targets_action_1

    # shape (n_features, 2)
    new_params = jnp.hstack((params_action_0, params_action_1))

    unfrozen_params = params.unfreeze()
    # shape (2, n_features, 2)
    unfrozen_params["params"]["Dense_0"]["kernel"] = jnp.repeat(new_params[None], 2, axis=0)

    return FrozenDict(unfrozen_params)
