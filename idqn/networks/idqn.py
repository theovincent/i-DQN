from typing import Tuple
from flax.core import FrozenDict
from functools import partial
import flax.linen as nn
import jax
import jax.numpy as jnp

from idqn.networks.base import BaseIteratedQ
from idqn.networks.architectures.base import scale
from idqn.sample_collection import IDX_RB
from idqn.sample_collection.replay_buffer import ReplayBuffer


class iDQN(BaseIteratedQ):
    def __init__(
        self,
        n_heads: int,
        state_shape: list,
        n_actions: int,
        cumulative_gamma: float,
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
            cumulative_gamma,
            network,
            network_key,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
            n_training_steps_per_rolling_step,
        )
        self.head_behaviorial_probability = head_behaviorial_probability

    def apply(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply(params, scale(state))

    def apply_other_heads(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply_other_heads(params, scale(state))

    def compute_target(self, params: FrozenDict, sample: Tuple[jnp.ndarray]) -> jnp.ndarray:
        # remove the last head | output (n_heads - 1)
        max_next_q = jnp.max(self.apply(params, sample[IDX_RB["next_state"]])[:-1], axis=1)

        # mapping over the heads
        return sample[IDX_RB["reward"]] + (1 - sample[IDX_RB["terminal"]]) * self.cumulative_gamma * max_next_q

    def loss(
        self, params: FrozenDict, params_target: FrozenDict, sample: Tuple[jnp.ndarray], key: jax.random.PRNGKeyArray
    ) -> jnp.float32:
        # output (n_heads - 1)
        targets = self.compute_target(params_target, sample)
        # output (n_heads - 1)
        q_values = self.apply_other_heads(params, sample[IDX_RB["state"]])[:, sample[IDX_RB["action"]]]

        return self.metric(q_values - targets, ord="2")

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKeyArray) -> jnp.int8:
        idx_head = self.random_head(key, self.head_behaviorial_probability)

        return jnp.argmax(self.network.apply_specific_head(params, idx_head, scale(state))).astype(jnp.int8)

    def compute_standard_deviation_head(self, replay_buffer: ReplayBuffer, key: jax.random.PRNGKeyArray) -> jnp.float32:
        standard_deviation = 0

        for _ in range(100):
            batch_sample = replay_buffer.sample_random_batch(key)
            standard_deviation += jnp.std(self(self.params, batch_sample[IDX_RB["state"]]), axis=1).sum()

        return standard_deviation / (100 * replay_buffer.batch_size * self.n_actions)

    def compute_approximation_error(self, replay_buffer: ReplayBuffer, key: jax.random.PRNGKeyArray) -> jnp.float32:
        approximation_error = 0

        for _ in range(500):
            batch_sample = replay_buffer.sample_random_batch(key)

            targets = self.compute_target(self.target_params, batch_sample)[:, 0]
            values_actions = self.apply(self.params, batch_sample[IDX_RB["state"]])
            # mapping over the state
            predictions = jax.vmap(lambda value_actions, action: value_actions[1, action])(
                values_actions, batch_sample[IDX_RB["action"]]
            )
            approximation_error += self.metric(predictions - targets, ord="sum")

        return approximation_error / (500 * replay_buffer.batch_size)
