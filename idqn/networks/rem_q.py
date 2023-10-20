from flax.core import FrozenDict
from functools import partial
import flax.linen as nn
import jax
import jax.numpy as jnp

from idqn.networks.dqn_q import DQN
from idqn.sample_collection.replay_buffer import ReplayBuffer


class REM(DQN):
    def __init__(
        self,
        state_shape: list,
        n_actions: int,
        gamma: float,
        network: nn.Module,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
        epsilon_optimizer: float,
        n_training_steps_per_online_update: int,
        n_training_steps_per_target_update: int,
    ) -> None:
        super().__init__(
            state_shape,
            n_actions,
            gamma,
            network,
            network_key,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
        )

    @partial(jax.jit, static_argnames="self")
    def update_combination(self, params: FrozenDict, key: jax.random.PRNGKeyArray) -> FrozenDict:
        return self.network.update_combination(params, key)

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer, **kwargs) -> jnp.float32:
        if kwargs.get("has_reset"):
            self.network_key, key = jax.random.split(self.network_key)
            self.params = self.update_combination(self.params, key)

        return super().update_online_params(step, replay_buffer, **kwargs)
