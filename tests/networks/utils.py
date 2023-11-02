from typing import Tuple
from functools import partial
import jax
import jax.numpy as jnp


class Generator:
    def __init__(self, batch_size: int, state_shape: Tuple[int]) -> None:
        self.batch_size = batch_size
        self.state_shape = state_shape

    @partial(jax.jit, static_argnames="self")
    def generate_samples(self, key: jax.random.PRNGKeyArray) -> Tuple[jnp.ndarray]:
        states = jax.random.uniform(key, (self.batch_size,) + self.state_shape)
        actions = jax.random.uniform(key, (self.batch_size,))
        _, key_ = jax.random.split(key)
        rewards = jax.random.uniform(key_, (self.batch_size,))
        terminals = jax.random.randint(key_, (self.batch_size,), 0, 2)
        next_states = jax.random.uniform(key_, (self.batch_size,) + self.state_shape)
        return (
            jnp.array(states, dtype=jnp.float32),  # state
            jnp.array(actions, dtype=jnp.int8),  # action
            jnp.array(rewards, dtype=jnp.float32),  # reward
            jnp.array(next_states, dtype=jnp.float32),  # next_state
            jnp.ones((self.batch_size)),  # next_action
            jnp.ones((self.batch_size)),  # next_reward
            jnp.array(terminals, dtype=jnp.bool_),  # terminal
            jnp.ones((self.batch_size)),  # indices
        )

    @partial(jax.jit, static_argnames="self")
    def generate_sample(
        self,
        key: jax.random.PRNGKeyArray,
    ) -> Tuple[jnp.ndarray]:
        states = jax.random.uniform(key, self.state_shape)
        actions = jax.random.uniform(key)
        _, key_ = jax.random.split(key)
        rewards = jax.random.uniform(key_)
        terminals = jax.random.randint(key_, (), 0, 2)
        next_states = jax.random.uniform(key_, self.state_shape)
        return (
            jnp.array(states, dtype=jnp.float32),  # state
            jnp.array(actions, dtype=jnp.int8),  # action
            jnp.array(rewards, dtype=jnp.float32),  # reward
            jnp.array(next_states, dtype=jnp.float32),  # next_state
            jnp.ones(1),  # next_action
            jnp.ones(1),  # next_reward
            jnp.array(terminals, dtype=jnp.bool_),  # terminal
            jnp.ones(1),  # indices
        )

    @partial(jax.jit, static_argnames="self")
    def generate_states(self, key: jax.random.PRNGKeyArray) -> jnp.ndarray:
        return jax.random.uniform(key, (self.batch_size,) + self.state_shape)

    @partial(jax.jit, static_argnames="self")
    def generate_state(self, key: jax.random.PRNGKeyArray) -> jnp.ndarray:
        return jax.random.uniform(key, self.state_shape)
