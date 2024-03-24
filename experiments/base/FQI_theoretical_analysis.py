from tqdm import tqdm
from argparse import Namespace
import jax
import numpy as np

from idqn.sample_collection.replay_buffer import ReplayBuffer
from idqn.networks.architectures.ifqi import CarOnHilliFQI
from idqn.environments.car_on_hill import CarOnHillEnv


def train(
    experiment_path: str,
    args: Namespace,
    p: dict,
    q: CarOnHilliFQI,
    env: CarOnHillEnv,
    replay_buffer: ReplayBuffer,
) -> None:
    n_target_update = p["n_bellman_iterations"] // args.bellman_iterations_scope
    n_training_steps_per_target_update = p["n_training_steps_per_bellman_iterations"] * args.bellman_iterations_scope

    env.collect_random_samples(
        jax.random.PRNGKey(0),
        replay_buffer,
        p["n_random_samples"],
        p["n_oriented_samples"],
        p["oriented_states"],
        p["horizon"],
    )
    delayed_update_frequency = 0
    bound_info = []

    for _ in tqdm(range(n_target_update)):
        for _ in range(n_training_steps_per_target_update):
            # 1 so that a training step is done and no rolling step is done
            q.update_online_params(1, replay_buffer)

            proposition_value = compute_proposition_value(q, replay_buffer)

            if proposition_value >= 0:
                diff_approximation_errors = compute_diff_approximation_errors(q, replay_buffer)
                bound_info.append([delayed_update_frequency, proposition_value, diff_approximation_errors])

                # 0 so that the target parameters are updated
                q.update_target_params(0)
                delayed_update_frequency = 0
            else:
                delayed_update_frequency += 1

        q.rolling_step(q.params)
        q.rolling_step(q.target_params)

    np.save(
        f"{experiment_path}{args.bellman_iterations_scope}_bound_info_s{args.seed}.npy",
        np.array(bound_info),
    )
