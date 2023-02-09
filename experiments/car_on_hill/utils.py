from typing import Tuple
from jax.random import KeyArray
import numpy as np
import jax
import jax.numpy as jnp

from idqn.environments.car_on_hill import CarOnHillEnv
from idqn.networks.learnable_multi_head_q import FullyConnectedMultiQ
from idqn.sample_collection.replay_buffer import ReplayBuffer
from idqn.sample_collection.dataloader import SampleDataLoader


def define_environment(
    gamma: float, n_states_x: int, n_states_v: int
) -> Tuple[CarOnHillEnv, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    env = CarOnHillEnv(gamma)

    states_x = np.linspace(-env.max_position, env.max_position, n_states_x)
    boxes_x_size = (2 * env.max_position) / (n_states_x - 1)
    states_x_boxes = np.linspace(-env.max_position, env.max_position + boxes_x_size, n_states_x + 1) - boxes_x_size / 2
    states_v = np.linspace(-env.max_velocity, env.max_velocity, n_states_v)
    boxes_v_size = (2 * env.max_velocity) / (n_states_v - 1)
    states_v_boxes = np.linspace(-env.max_velocity, env.max_velocity + boxes_v_size, n_states_v + 1) - boxes_v_size / 2

    return env, states_x, states_x_boxes, states_v, states_v_boxes


def define_multi_q(
    importance_iteration: jnp.ndarray,
    gamma: float,
    key: jax.random.PRNGKeyArray,
    layers_dimension: dict,
    learning_rate: dict = None,
) -> FullyConnectedMultiQ:
    return FullyConnectedMultiQ(
        importance_iteration=importance_iteration,
        state_shape=(2,),
        n_actions=2,
        gamma=gamma,
        network_key=key,
        layers_dimension=layers_dimension,
        zero_initializer=True,
        learning_rate=learning_rate,
    )


def define_data_loader_samples(n_samples, experiment_name: str, batch_size_samples, key) -> SampleDataLoader:
    replay_buffer = ReplayBuffer(n_samples)
    replay_buffer.load(f"experiments/car_on_hill/figures/{experiment_name}/replay_buffer.npz")
    return SampleDataLoader(replay_buffer, batch_size_samples, key)


def generate_keys(seed: int) -> KeyArray:
    return jax.random.split(jax.random.PRNGKey(seed), 2)
