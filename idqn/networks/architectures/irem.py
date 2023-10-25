from typing import Callable
from functools import partial
import flax.linen as nn
from flax.core import FrozenDict
import jax
import jax.numpy as jnp

from idqn.networks.irem import iREM
from idqn.networks.architectures.base import roll
from idqn.networks.architectures.idqn import AtariSharediDQNNet, AtariiDQNNet


class AtariiREMNet:
    def __init__(self, n_nets_rem: int, shared_network: bool, n_head_idqn: int, n_actions: int) -> None:
        self.n_nets_rem = n_nets_rem
        self.idqn_net = (
            AtariSharediDQNNet(n_head_idqn, n_actions) if shared_network else AtariiDQNNet(n_head_idqn, n_actions)
        )

        uniform_initializer = jax.nn.initializers.uniform(1)
        normed_uniform_initializer = lambda *args: uniform_initializer(*args) / uniform_initializer(*args).sum()
        self.combiner = nn.Dense(features=1, kernel_init=normed_uniform_initializer, use_bias=False)

    def init(self, key_init: jax.random.PRNGKey, state: jnp.ndarray) -> FrozenDict:
        params_nets = jax.vmap(self.idqn_net.init, in_axes=(0, None))(
            jax.random.split(key_init, self.n_nets_rem), state
        )
        params_combiner = self.combiner.init(key_init, jnp.ones(self.n_nets_rem))

        return FrozenDict(params_nets=params_nets, params_combiner=params_combiner)

    def apply_generic(self, idqn_apply_func: Callable, params: FrozenDict, state: jnp.ndarray):
        # out_axes = -1 because the n_nets networks are the input dimension of the combiner
        # output (batch_size, n_head_idqn, n_actions, n_nets_rem)
        output_nets = jax.vmap(
            lambda params, state: idqn_apply_func(params, state=state), in_axes=(0, None), out_axes=-1
        )(params["params_nets"], state)
        params_combiner = jax.lax.stop_gradient(params["params_combiner"])

        # squeeze last dimension because the combination of the n_nets networks should not appear
        # output (batch_size, n_head_idqn, n_actions)
        return jnp.squeeze(self.combiner.apply(params_combiner, output_nets), axis=-1)

    def apply(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        return self.apply_generic(self.idqn_net.apply, params, state)

    def apply_other_heads(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        return self.apply_generic(self.idqn_net.apply_other_heads, params, state)

    def apply_specific_head(self, params: FrozenDict, idx_head: int, state: jnp.ndarray) -> jnp.ndarray:
        return self.apply_generic(partial(self.idqn_net.apply_specific_head, idx_head=idx_head), params, state)

    def rolling_step(self, params: FrozenDict) -> FrozenDict:
        unfreezed_params = params.unfreeze()
        unfreezed_params["params_nets"] = jax.vmap(
            lambda params: jax.tree_util.tree_map(lambda param: roll(param), params)
        )(unfreezed_params["params_nets"])

        return FrozenDict(unfreezed_params)

    def update_combination(self, params: FrozenDict, key: jax.random.PRNGKeyArray) -> FrozenDict:
        unfrozen_params = params.unfreeze()
        unfrozen_params["params_combiner"] = self.combiner.init(key, jnp.ones(self.n_nets_rem))

        return FrozenDict(unfrozen_params)


class AtariiREM(iREM):
    def __init__(
        self,
        n_heads: int,
        state_shape: list,
        n_actions: int,
        gamma: float,
        network_key: jax.random.PRNGKeyArray,
        head_behaviorial_probability: jnp.ndarray,
        learning_rate: float,
        epsilon_optimizer: float,
        n_training_steps_per_online_update: int,
        n_training_steps_per_target_update: int,
        n_training_steps_per_rolling_step: int,
        shared_network: int,
    ) -> None:
        super().__init__(
            n_heads,
            state_shape,
            n_actions,
            gamma,
            AtariiREMNet(4, shared_network, n_heads, n_actions),
            network_key,
            head_behaviorial_probability,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
            n_training_steps_per_rolling_step,
        )
