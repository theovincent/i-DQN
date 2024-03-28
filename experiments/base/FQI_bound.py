from tqdm import tqdm
from argparse import Namespace
import jax
import jax.numpy as jnp
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
    list_parameters = [jax.tree_map(lambda param: jnp.repeat(param[0][None], 2, axis=0), q.target_params)]

    for _ in tqdm(range(n_rolling_steps)):
        for _ in range(n_training_steps_per_rolling_step):
            # 1 so that a training step is done and no rolling step is done
            q.update_online_params(1, replay_buffer)
            n_training_steps += 1

            if n_training_steps % 30 == 0:
                proposition_value = q.compute_proposition_value(q.params, q.target_params, dataset, states, p["gamma"])
                diff_approximation_errors = q.compute_diff_approximation_errors(q.params, q.target_params, dataset)

                bound_info.append([proposition_value, diff_approximation_errors])

                # 0 so that the target parameters are updated
                q.update_target_params(0)

        list_parameters.append(jax.tree_map(lambda param: jnp.repeat(param[1][None], 2, axis=0), q.params))

        q.params = q.rolling_step(q.params)
        q.target_params = q.rolling_step(q.target_params)

    np.save(
        f"{experiment_path}{args.bellman_iterations_scope}_bound_info_s{args.seed}",
        np.array(bound_info),
    )

    for idx_in_params in range(2, args.bellman_iterations_scope + 1):
        list_parameters.append(jax.tree_map(lambda param: jnp.repeat(param[idx_in_params][None], 2, axis=0), q.params))

    distance_optimal_q = np.zeros(p["n_bellman_iterations"] + 1) * np.nan
    approximation_errors = np.zeros(p["n_bellman_iterations"] + 1) * np.nan
    optimal_q_values = np.load("experiments/car_on_hill/figures/data/optimal/Q.npy")
    samples_mask = np.load("experiments/car_on_hill/figures/data/samples_count.npy")
    samples_mask_q_format = np.repeat(samples_mask[:, :, None], 2, axis=-1)
    states_x = np.linspace(-env.max_position, env.max_position, p["n_states_x"])
    states_v = np.linspace(-env.max_velocity, env.max_velocity, p["n_states_v"])

    for idx_bellman_iteration in range(1, p["n_bellman_iterations"] + 1):
        q_values = env.q_estimate_mesh(
            q,
            jax.tree_map(lambda param: param[1][None], list_parameters[idx_bellman_iteration]),
            states_x,
            states_v,
        )
        distance_optimal_q[idx_bellman_iteration] = (
            np.sqrt(np.square((optimal_q_values - q_values) * samples_mask_q_format).mean())
            / dataset[0].shape[0]
            * p["n_states_x"]
            * p["n_states_v"]
        )
        approximation_errors[idx_bellman_iteration] = q.loss_on_batch(
            list_parameters[idx_bellman_iteration], list_parameters[idx_bellman_iteration - 1], dataset, None
        )

    np.save(f"{experiment_path}{args.bellman_iterations_scope}_distance_Q_s{args.seed}", distance_optimal_q)
    np.save(f"{experiment_path}{args.bellman_iterations_scope}_approx_errors_s{args.seed}", approximation_errors)
