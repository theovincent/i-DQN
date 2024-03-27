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
    n_rolling_steps = p["n_bellman_iterations"] - args.bellman_iterations_scope + 1
    n_training_steps_per_rolling_step = (
        p["n_training_steps_per_bellman_iterations"] * p["n_bellman_iterations"] // n_rolling_steps
    )

    env.collect_random_samples(
        jax.random.PRNGKey(0),
        replay_buffer,
        p["n_random_samples"],
        p["n_oriented_samples"],
        p["oriented_states"],
        p["horizon"],
    )

    n_training_steps = 0
    dataset = replay_buffer.get_all_valid_samples()
    states = env.grid_states(200)
    bound_info = []

    for idx_rolling_step in tqdm(range(n_rolling_steps)):
        for _ in range(n_training_steps_per_rolling_step):
            # 1 so that a training step is done and no rolling step is done
            q.update_online_params(1, replay_buffer)
            n_training_steps += 1

            if n_training_steps % 1 == 0:
                # proposition_value = q.compute_proposition_value(q.params, q.target_params, dataset, states, p["gamma"])
                # diff_approximation_errors = q.compute_diff_approximation_errors(q.params, q.target_params, dataset)

                # bound_info.append([proposition_value, diff_approximation_errors])

                # 0 so that the target parameters are updated
                q.update_target_params(0)

        q.save(f"{experiment_path}{args.bellman_iterations_scope}_Q_s{args.seed}_r{idx_rolling_step}")

        q.params = q.rolling_step(q.params)
        q.target_params = q.rolling_step(q.target_params)

        np.save(
            f"{experiment_path}{args.bellman_iterations_scope}_bound_info_s{args.seed}.npy",
            np.array(bound_info),
        )
