from typing import Tuple
from flax.core import FrozenDict
from functools import partial
import flax.linen as nn
import jax
import jax.numpy as jnp

from idqn.networks.base import BaseSingleQ
from idqn.networks.architectures.base import scale
from idqn.sample_collection import IDX_RB


class DQN(BaseSingleQ):
    def __init__(
        self,
        state_shape: list,
        n_actions: int,
        cumulative_gamma: float,
        network: nn.Module,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
        epsilon_optimizer: float,
        n_training_steps_per_online_update: int,
        n_training_steps_per_target_update: int,
    ) -> None:
        super().__init__(
            {"state": jnp.zeros(state_shape, dtype=jnp.float32)},
            n_actions,
            cumulative_gamma,
            network,
            network_key,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
        )

    def apply(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply(params, scale(state))

    def compute_target(self, params: FrozenDict, sample: Tuple[jnp.ndarray]) -> jnp.float32:
        return sample[IDX_RB["reward"]] + (1 - sample[IDX_RB["terminal"]]) * self.cumulative_gamma * jnp.max(
            self.apply(params, sample[IDX_RB["next_state"]])
        )

    def loss(
        self, params: FrozenDict, params_target: FrozenDict, sample: Tuple[jnp.ndarray], key: jax.random.PRNGKeyArray
    ) -> jnp.float32:
        target = self.compute_target(params_target, sample)
        q_value = self.apply(params, sample[IDX_RB["state"]])[sample[IDX_RB["action"]]]

        return self.metric(q_value - target, ord="2")

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKeyArray) -> jnp.int8:
        return jnp.argmax(self.network.apply(params, scale(state))).astype(jnp.int8)
