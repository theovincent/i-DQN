from typing import Tuple
from flax.core import FrozenDict
from functools import partial
import flax.linen as nn
import jax
import jax.numpy as jnp

from idqn.networks.base import BaseSingleQ
from idqn.networks.architectures.base import scale, preprocessor
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

    def apply(self, params: FrozenDict, states: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self.network.apply, in_axes=(None, 0))(params, preprocessor(states))

    def compute_target(self, params: FrozenDict, samples: Tuple[jnp.ndarray]) -> jnp.ndarray:
        return samples[IDX_RB["reward"]] + (1 - samples[IDX_RB["terminal"]]) * self.cumulative_gamma * self.apply(
            params, samples[IDX_RB["next_state"]]
        ).max(axis=1)

    def loss(self, params: FrozenDict, params_target: FrozenDict, samples: Tuple[jnp.ndarray]) -> jnp.float32:
        targets = self.compute_target(params_target, samples)
        q_states_actions = self.apply(params, samples[IDX_RB["state"]])

        # mapping over the states
        predictions = jax.vmap(lambda q_state_actions, action: q_state_actions[action])(
            q_states_actions, samples[IDX_RB["action"]]
        )

        return self.metric(predictions - targets, ord="2")

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, **kwargs) -> jnp.int8:
        return jnp.argmax(self.network.apply(params, scale(state))).astype(jnp.int8)
