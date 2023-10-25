from flax.core import FrozenDict
import jax
import jax.numpy as jnp

from idqn.networks.idqn import iDQN
from idqn.networks.architectures.base import Torso, Head, roll
from idqn.networks.architectures.dqn import AtariDQNNet


class AtariSharediDQNNet:
    def __init__(self, n_heads: int, n_actions: int) -> None:
        self.n_heads = n_heads
        self.n_actions = n_actions
        self.torso = Torso()
        self.head = Head(self.n_actions)

    def init(self, key_init: jax.random.PRNGKey, state: jnp.ndarray) -> FrozenDict:
        # We need only two sets of torso parameters
        torso_params = jax.vmap(self.torso.init, in_axes=(0, None))(jax.random.split(key_init), state)

        features = self.torso.apply(jax.tree_util.tree_map(lambda param: param[0], torso_params), state)

        head_params = jax.vmap(self.head.init, in_axes=(0, None))(jax.random.split(key_init, self.n_heads), features)

        return FrozenDict(torso_params=torso_params, head_params=head_params)

    def apply(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        # output (2, batch_size, n_features)
        features = jax.vmap(self.torso.apply, in_axes=(0, None))(params["torso_params"], state)

        # batch_size = features_0.shape[1]
        output = jnp.zeros((features.shape[1], self.n_heads, self.n_actions))

        output = output.at[:, 0].set(
            self.head.apply(jax.tree_util.tree_map(lambda param: param[0], params["head_params"]), features[0])
        )
        output = output.at[:, 1:].set(
            jax.vmap(self.head.apply, in_axes=(0, None), out_axes=1)(
                jax.tree_util.tree_map(lambda param: param[1:], params["head_params"]), features[1]
            )  # output (batch_size, n_heads - 1, n_features)
        )

        return output  # output (batch_size, n_heads, n_features)

    def apply_other_heads(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        feature = self.torso.apply(jax.tree_util.tree_map(lambda param: param[1], params["torso_params"]), state)

        return jax.vmap(self.head.apply, in_axes=(0, None), out_axes=1)(
            jax.tree_util.tree_map(lambda param: param[1:], params["head_params"]), feature
        )  # output (batch_size, n_heads - 1, n_features)

    def apply_specific_head(self, params: FrozenDict, idx_head: int, state: jnp.ndarray) -> jnp.ndarray:
        feature = self.torso.apply(
            jax.tree_util.tree_map(
                lambda param: param[jax.lax.cond(idx_head >= 1, lambda: 1, lambda: 0)], params["torso_params"]
            ),
            state,
        )

        return self.head.apply(jax.tree_util.tree_map(lambda param: param[idx_head], params["head_params"]), feature)

    def rolling_step(self, params: FrozenDict) -> FrozenDict:
        return FrozenDict(jax.tree_util.tree_map(lambda param: roll(param), params.unfreeze()))


class AtariiDQNNet:
    def __init__(self, n_nets: int, n_actions: int) -> None:
        self.n_nets = n_nets
        self.dqn_net = AtariDQNNet(n_actions)

    def init(self, key_init: jax.random.PRNGKey, state: jnp.ndarray) -> FrozenDict:
        return jax.vmap(self.dqn_net.init, in_axes=(0, None))(jax.random.split(key_init, self.n_nets), state)

    def apply(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        # output (batch_size, n_nets, n_actions)
        return jax.vmap(self.dqn_net.apply, in_axes=(0, None), out_axes=1)(params, state)

    def apply_other_heads(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        params_other_heads = jax.tree_util.tree_map(lambda param: param[1:], params)

        return self.apply(params_other_heads, state)

    def apply_specific_head(self, params: FrozenDict, idx_head: int, state: jnp.ndarray) -> jnp.ndarray:
        params_heads = jax.tree_util.tree_map(lambda param: param[idx_head], params)

        return self.dqn_net.apply(params_heads, state)

    def rolling_step(self, params: FrozenDict) -> FrozenDict:
        return FrozenDict(jax.tree_util.tree_map(lambda param: roll(param), params.unfreeze()))


class AtariiDQN(iDQN):
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
            AtariSharediDQNNet(n_heads, n_actions) if shared_network else AtariiDQNNet(n_heads, n_actions),
            network_key,
            head_behaviorial_probability,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
            n_training_steps_per_rolling_step,
        )
