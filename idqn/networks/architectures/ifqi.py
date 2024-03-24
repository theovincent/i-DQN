from typing import Sequence
from flax.core import FrozenDict
import jax
import jax.numpy as jnp

from idqn.networks.idqn import iDQN
from idqn.networks.architectures.base import MLP, roll


class CarOnHilliFQINet:
    def __init__(self, n_nets: int, features: Sequence, n_actions: int) -> None:
        self.n_nets = n_nets
        self.fqi_net = MLP(features, n_actions)

    def init(self, key_init: jax.random.PRNGKey, state: jnp.ndarray) -> FrozenDict:
        # Initialize each network at the same location
        return jax.vmap(self.fqi_net.init, in_axes=(0, None))(jnp.repeat(key_init[None], self.n_nets, axis=0), state)

    def apply(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        # output (n_nets, n_actions)
        return jax.vmap(self.fqi_net.apply, in_axes=(0, None))(params, state)

    def apply_other_heads(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        params_other_heads = jax.tree_util.tree_map(lambda param: param[1:], params)

        return self.apply(params_other_heads, state)

    def apply_specific_head(self, params: FrozenDict, idx_head: int, state: jnp.ndarray) -> jnp.ndarray:
        params_heads = jax.tree_util.tree_map(lambda param: param[idx_head], params)

        return self.dqn_net.apply(params_heads, state)

    def rolling_step(self, params: FrozenDict) -> FrozenDict:
        return FrozenDict(jax.tree_util.tree_map(lambda param: roll(param), params.unfreeze()))


class CarOnHilliFQI(iDQN):
    def __init__(
        self,
        n_heads: int,
        features: Sequence,
        state_shape: list,
        n_actions: int,
        cumulative_gamma: float,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
        epsilon_optimizer: float,
    ) -> None:
        super().__init__(
            n_heads,
            state_shape,
            n_actions,
            cumulative_gamma,
            CarOnHilliFQINet(n_heads, features, n_actions),
            network_key,
            None,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update=1,  # always train
            n_training_steps_per_target_update=1,  # always update the target
            n_training_steps_per_rolling_step=1e10,  # never roll the params
        )

    def apply(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply(params, state)

    def apply_other_heads(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply_other_heads(params, state)
