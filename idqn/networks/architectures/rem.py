import flax.linen as nn
from flax.core import FrozenDict
import jax
import jax.numpy as jnp

from idqn.networks.rem import REM
from idqn.networks.architectures.dqn import AtariDQNNet


class AtariREMNet:
    def __init__(self, n_nets: int, n_actions: int) -> None:
        self.n_nets = n_nets
        self.dqn_net = AtariDQNNet(n_actions)

        uniform_initializer = jax.nn.initializers.uniform(1)
        normed_uniform_initializer = lambda *args: uniform_initializer(*args) / uniform_initializer(*args).sum()
        self.combiner = nn.Dense(features=1, kernel_init=normed_uniform_initializer, use_bias=False)

    def init(self, key_init: jax.random.PRNGKey, state: jnp.ndarray) -> FrozenDict:
        params_nets = jax.vmap(self.dqn_net.init, in_axes=(0, None))(jax.random.split(key_init, self.n_nets), state)
        params_combiner = self.combiner.init(key_init, jnp.ones(self.n_nets))

        return FrozenDict(params_nets=params_nets, params_combiner=params_combiner)

    def apply(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        # out_axes = -1 because the n_nets networks are the input dimension of the combiner
        output_nets = jax.vmap(self.dqn_net.apply, in_axes=(0, None), out_axes=-1)(params["params_nets"], state)
        params_combiner = jax.lax.stop_gradient(params["params_combiner"])

        # squeeze last dimension because the combination of the n_nets networks should not appear
        return jnp.squeeze(self.combiner.apply(params_combiner, output_nets), axis=-1)

    def update_combination(self, params: FrozenDict, key: jax.random.PRNGKeyArray) -> FrozenDict:
        unfrozen_params = params.unfreeze()
        unfrozen_params["params_combiner"] = self.combiner.init(key, jnp.ones(self.n_nets))

        return FrozenDict(unfrozen_params)


class AtariREM(REM):
    def __init__(
        self,
        state_shape: list,
        n_actions: int,
        gamma: float,
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
            AtariREMNet(4, n_actions),
            network_key,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
        )
