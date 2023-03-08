from tqdm import tqdm
from jax.random import KeyArray
import jax
import jax.numpy as jnp
import haiku as hk

from idqn.environments.atari import AtariEnv
from idqn.sample_collection.exploration import EpsilonGreedySchedule
from idqn.networks.base_q import BaseMultiHeadQ
from idqn.networks.learnable_multi_head_q import AtariMultiQ
from idqn.sample_collection.replay_buffer import ReplayBuffer


def define_environment(env_key: jax.random.PRNGKeyArray, name: str, gamma: float) -> AtariEnv:
    env = AtariEnv(env_key, name, gamma, terminal_on_life_loss=False)

    return env


def define_multi_q(
    importance_iteration: jnp.ndarray,
    state_shape: list,
    n_actions: int,
    gamma: float,
    key: jax.random.PRNGKeyArray,
    n_shared_layers: int,
    learning_rate: float = None,
) -> AtariMultiQ:
    return AtariMultiQ(
        importance_iteration=importance_iteration,
        state_shape=state_shape,
        n_actions=n_actions,
        gamma=gamma,
        network_key=key,
        n_shared_layers=n_shared_layers,
        zero_initializer=True,
        learning_rate=learning_rate,
    )


def collect_random_samples(
    env: AtariEnv,
    sample_key: jax.random.PRNGKeyArray,
    replay_buffer: ReplayBuffer,
    n_initial_samples: int,
    horizon: int,
) -> None:
    env.reset()

    for _ in tqdm(range(n_initial_samples)):
        state = env.state

        sample_key, key = jax.random.split(sample_key)
        action = jax.random.choice(key, jnp.arange(env.n_actions))
        next_state, reward, absorbing, _ = env.step(action)

        replay_buffer.add(state, action, reward, next_state, absorbing)

        if absorbing or env.n_steps >= horizon:
            env.reset()


def collect_samples_multi_head(
    env: AtariEnv,
    replay_buffer: ReplayBuffer,
    q: BaseMultiHeadQ,
    q_params: hk.Params,
    n_steps: int,
    horizon: int,
    exploration_schedule: EpsilonGreedySchedule,
    head_behaviorial_probability: jnp.ndarray,
) -> None:
    for _ in range(n_steps):
        state = env.state

        if exploration_schedule.explore():
            action = env.random_action(exploration_schedule.key)
        else:
            idx_head = env.random_head(exploration_schedule.key, q.n_heads, head_behaviorial_probability)
            action = env.best_action_multi_head(q, idx_head, q_params, state)

        next_state, reward, absorbing, _ = env.step(action)

        replay_buffer.add(state, action, reward, next_state, absorbing)

        if absorbing or env.n_steps >= horizon:
            env.reset()


def generate_keys(seed: int) -> KeyArray:
    return jax.random.split(jax.random.PRNGKey(seed), 3)
