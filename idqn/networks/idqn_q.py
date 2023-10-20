from typing import Tuple
from flax.core import FrozenDict
from functools import partial
import flax.linen as nn
import jax
import jax.numpy as jnp

from idqn.networks.base_q import BaseIteratedQ
from idqn.sample_collection import IDX_RB
from idqn.sample_collection.replay_buffer import ReplayBuffer


class iDQN(BaseIteratedQ):
    def __init__(
        self,
        n_heads: int,
        state_shape: list,
        n_actions: int,
        gamma: float,
        network: nn.Module,
        network_key: jax.random.PRNGKeyArray,
        head_behaviorial_probability: jnp.ndarray,
        learning_rate: float,
        epsilon_optimizer: float,
        n_training_steps_per_online_update: int,
        n_training_steps_per_target_update: int,
        n_training_steps_per_rolling_step: int,
    ) -> None:
        super().__init__(
            n_heads,
            {"state": jnp.zeros(state_shape, dtype=jnp.float32)},
            n_actions,
            gamma,
            network,
            network_key,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
            n_training_steps_per_rolling_step,
        )
        self.head_behaviorial_probability = head_behaviorial_probability

    @partial(jax.jit, static_argnames="self")
    def apply(self, params: FrozenDict, states: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply(params, states)

    @partial(jax.jit, static_argnames="self")
    def apply_other_heads(self, params: FrozenDict, states: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply_other_heads(params, states)

    @partial(jax.jit, static_argnames="self")
    def best_action_from_head(
        self, torso_params: FrozenDict, head_params: FrozenDict, state: jnp.ndarray
    ) -> jnp.ndarray:
        """This function is supposed to take a single state and not a batch"""
        return jnp.argmax(self.network.apply_specific_head(torso_params, head_params, state)[0]).astype(jnp.int8)

    @partial(jax.jit, static_argnames="self")
    def compute_target(self, params: FrozenDict, samples: Tuple[jnp.ndarray]) -> jnp.ndarray:
        # mapping over the states
        return jax.vmap(
            lambda reward, terminal, max_next_states: reward + (1 - terminal) * self.gamma * max_next_states,
        )(
            samples[IDX_RB["reward"]],
            samples[IDX_RB["terminal"]],
            jnp.max(self.apply(params, samples[IDX_RB["next_state"]]), axis=2),
        )

    @partial(jax.jit, static_argnames="self")
    def loss(self, params: FrozenDict, params_target: FrozenDict, samples: Tuple[jnp.ndarray]) -> jnp.float32:
        targets = self.compute_target(params_target, samples)[:, :-1]
        values_actions = self.apply_other_heads(params, samples[IDX_RB["state"]])
        # mapping over the states
        predictions = jax.vmap(lambda value_actions, action: value_actions[:, action])(
            values_actions, samples[IDX_RB["action"]]
        )

        return self.metric(predictions - targets, ord="2")

    def best_action(self, params: FrozenDict, state: jnp.ndarray, **kwargs) -> jnp.int8:
        idx_head = self.random_head(kwargs.get("key"), self.head_behaviorial_probability)

        return self.best_action_from_head(
            params[f"torso_params_{0 if idx_head == 0 else 1}"], params[f"head_params_{idx_head}"], state
        )

    def compute_standard_deviation_head(self, replay_buffer: ReplayBuffer, key: jax.random.PRNGKeyArray) -> jnp.float32:
        standard_deviation = 0

        for _ in range(100):
            batch_samples = replay_buffer.sample_random_batch(key)
            standard_deviation += jnp.std(self(self.params, batch_samples[IDX_RB["state"]]), axis=1).sum()

        return standard_deviation / (100 * replay_buffer.batch_size * self.n_actions)

    def compute_approximation_error(self, replay_buffer: ReplayBuffer, key: jax.random.PRNGKeyArray) -> jnp.float32:
        approximation_error = 0

        for _ in range(500):
            batch_samples = replay_buffer.sample_random_batch(key)

            targets = self.compute_target(self.target_params, batch_samples)[:, 0]
            values_actions = self.apply(self.params, batch_samples[IDX_RB["state"]])
            # mapping over the states
            predictions = jax.vmap(lambda value_actions, action: value_actions[1, action])(
                values_actions, batch_samples[IDX_RB["action"]]
            )
            approximation_error += self.metric(predictions - targets, ord="sum")

        return approximation_error / (500 * replay_buffer.batch_size)
