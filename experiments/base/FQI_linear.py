from tqdm import tqdm
from argparse import Namespace
import jax
import jax.numpy as jnp
import numpy as np

from idqn.sample_collection.replay_buffer import ReplayBuffer
from idqn.networks.architectures.ifqi import CarOnHilliFQI
from idqn.environments.car_on_hill import CarOnHillEnv
from experiments.car_on_hill.iFQI_linear_closed_form import update_bellman_iteration


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
    dataset = replay_buffer.get_all_valid_samples()
    list_parameters = [jax.tree_map(lambda param: jnp.repeat(param[0][None], 2, axis=0), q.target_params)]

    for _ in tqdm(range(1, n_rolling_steps + 1)):
        for _ in range(n_training_steps_per_rolling_step):
            # 1 so that a training step is done and no rolling step is done
            q.update_online_params(1, replay_buffer)
            q.update_target_params(0)

        # q.save(f"{experiment_path}{args.bellman_iterations_scope}_Q_s{args.seed}_r{idx_rolling_step}")

        list_parameters.append(jax.tree_map(lambda param: jnp.repeat(param[1][None], 2, axis=0), q.params))

        q.params = q.rolling_step(q.params)
        q.target_params = q.rolling_step(q.target_params)

    for idx_in_params in range(2, args.bellman_iterations_scope + 1):
        list_parameters.append(jax.tree_map(lambda param: jnp.repeat(param[idx_in_params][None], 2, axis=0), q.params))

    diff_weights = np.zeros(p["n_bellman_iterations"] + 1) * np.nan
    distance_optimal_q = np.zeros(p["n_bellman_iterations"] + 1) * np.nan
    optimal_q_values = np.load("experiments/car_on_hill/figures/data/optimal/Q.npy")
    samples_mask = np.load("experiments/car_on_hill/figures/data/samples_count.npy")
    samples_mask_q_format = np.repeat(samples_mask[:, :, None], 2, axis=-1)
    states_x = np.linspace(-env.max_position, env.max_position, p["n_states_x"])
    states_v = np.linspace(-env.max_velocity, env.max_velocity, p["n_states_v"])

    for idx_bellman_iteration in range(1, p["n_bellman_iterations"] + 1):
        optimal_params = update_bellman_iteration(q, list_parameters[idx_bellman_iteration - 1], dataset)

        weights = list_parameters[idx_bellman_iteration]["params"]["Dense_0"]["kernel"][1]
        optimal_weights = optimal_params["params"]["Dense_0"]["kernel"][1]

        diff_weights[idx_bellman_iteration] = np.sqrt(np.square(weights - optimal_weights).mean())

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

    np.save(f"{experiment_path}{args.bellman_iterations_scope}_distance_w_s{args.seed}", diff_weights)
    np.save(f"{experiment_path}{args.bellman_iterations_scope}_distance_Q_s{args.seed}", distance_optimal_q)
