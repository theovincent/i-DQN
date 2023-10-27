from typing import Tuple
from flax.core import FrozenDict
import jax
import jax.numpy as jnp

from idqn.networks.iiqn import iIQN
from idqn.networks.architectures.base import Torso, Head, QuantileEmbedding, roll
from idqn.networks.architectures.iqn import AtariIQNNet


class AtariSharediIQNet:
    def __init__(self, n_heads: int, n_actions: int) -> None:
        self.n_heads = n_heads
        self.n_actions = n_actions
        self.torso = Torso(dqn_initialisation=False)
        self.quantile_embedding = QuantileEmbedding()
        self.head = Head(self.n_actions, dqn_initialisation=False)

    def init(
        self, key_init: jax.random.PRNGKey, state: jnp.ndarray, key: jax.random.PRNGKey, n_quantiles: int
    ) -> FrozenDict:
        # We need two sets of torso parameters
        torso_params = jax.vmap(self.torso.init, in_axes=(0, None))(jax.random.split(key_init), state)

        # We need two set of quantile embedding parameters. 1 for batch_size
        key_init, _ = jax.random.split(key_init)
        quantiles_params = jax.vmap(self.quantile_embedding.init, in_axes=(0, None, None, None))(
            jax.random.split(key_init), key, n_quantiles, 1
        )

        # Compute the input for the heads
        state_features = self.torso.apply(
            jax.tree_util.tree_map(lambda param: param[0], torso_params), state
        )  # output (batch_size, n_features)
        quantiles_features, _ = self.quantile_embedding.apply(
            jax.tree_util.tree_map(lambda param: param[0], quantiles_params), key, n_quantiles, 1
        )  # output (batch_size, n_quantiles, n_features)

        # mapping over the quantiles. output (batch_size, n_quantiles, n_features)
        multiplied_features = jax.vmap(
            lambda quantile_features, state_features_: quantile_features * state_features_,
            in_axes=(1, None),
            out_axes=1,
        )(quantiles_features, state_features)

        head_params = jax.vmap(self.head.init, in_axes=(0, None))(
            jax.random.split(key_init, self.n_heads), multiplied_features
        )

        return FrozenDict(torso_params=torso_params, quantiles_params=quantiles_params, head_params=head_params)

    def apply(
        self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKey, n_quantiles: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # output (2, batch_size, n_features)
        state_features = jax.vmap(self.torso.apply, in_axes=(0, None))(params["torso_params"], state)

        # We use the same key for the quantiles here so that the 'apply' function computes the same quantiles over all the heads.
        quantiles_features, quantiles = jax.vmap(self.quantile_embedding.apply, in_axes=(0, None, None, None))(
            params["quantiles_params"], key, n_quantiles, state_features.shape[1]
        )  # output (2, batch_size, n_quantiles, n_features) | (2, batch_size, n_quantiles)

        # mapping over the quantiles. output (2, batch_size, n_quantiles, n_features)
        multiplied_features = jax.vmap(
            lambda quantile_features_, state_features_: quantile_features_ * state_features_, (2, None), 2
        )(quantiles_features, state_features)

        output = jnp.zeros((state_features.shape[1], self.n_heads, n_quantiles, self.n_actions))

        output = output.at[:, 0].set(
            self.head.apply(
                jax.tree_util.tree_map(lambda param: param[0], params["head_params"]), multiplied_features[0]
            )  # output (batch_size, n_quantiles, n_actions)
        )
        output = output.at[:, 1:].set(
            jax.vmap(self.head.apply, in_axes=(0, None), out_axes=1)(
                jax.tree_util.tree_map(lambda param: param[1:], params["head_params"]), multiplied_features[1]
            )  # output (batch_size, n_heads - 1, n_quantiles, n_actions)
        )

        return output, quantiles[0]  # output (batch_size, n_heads, n_quantiles, n_actions) | (batch_size, n_quantiles)

    def apply_other_heads(
        self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKey, n_quantiles: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # output (batch_size, n_features)
        state_feature = self.torso.apply(jax.tree_util.tree_map(lambda param: param[1], params["torso_params"]), state)

        # output (batch_size, n_quantiles, n_features) | (batch_size, n_quantiles)
        quantiles_feature, quantiles = self.quantile_embedding.apply(
            jax.tree_util.tree_map(lambda param: param[1], params["quantiles_params"]),
            key,
            n_quantiles,
            state_feature.shape[0],
        )

        # mapping over the quantiles. output (batch_size, n_quantiles, n_features)
        multiplied_feature = jax.vmap(
            lambda quantile_feature_, state_feature_: quantile_feature_ * state_feature_, (1, None), 1
        )(quantiles_feature, state_feature)

        output = jax.vmap(self.head.apply, in_axes=(0, None), out_axes=1)(
            jax.tree_util.tree_map(lambda param: param[1:], params["head_params"]), multiplied_feature
        )  # output (batch_size, n_heads, n_quantiles, n_actions)

        return output, quantiles  # output (batch_size, n_heads, n_quantiles, n_actions) | (batch_size, n_quantiles)

    def apply_specific_head(
        self, params: FrozenDict, idx_head: int, state: jnp.ndarray, key: jax.random.PRNGKey, n_quantiles: int
    ) -> jnp.ndarray:
        # output (batch_size, n_features)
        state_feature = self.torso.apply(
            jax.tree_util.tree_map(
                lambda param: param[jax.lax.cond(idx_head >= 1, lambda: 1, lambda: 0)], params["torso_params"]
            ),
            state,
        )

        # output (batch_size, n_quantiles, n_features)
        quantiles_feature, _ = self.quantile_embedding.apply(
            jax.tree_util.tree_map(
                lambda param: param[jax.lax.cond(idx_head >= 1, lambda: 1, lambda: 0)], params["quantiles_params"]
            ),
            key,
            n_quantiles,
            state_feature.shape[0],
        )

        # mapping over the quantiles. output (batch_size, n_quantiles, n_features)
        multiplied_feature = jax.vmap(
            lambda quantile_feature_, state_feature_: quantile_feature_ * state_feature_, (1, None), 1
        )(quantiles_feature, state_feature)

        return self.head.apply(
            jax.tree_util.tree_map(lambda param: param[idx_head], params["head_params"]), multiplied_feature
        )  # output (batch_size, n_quantiles, n_actions)

    def rolling_step(self, params: FrozenDict) -> FrozenDict:
        return FrozenDict(jax.tree_util.tree_map(lambda param: roll(param), params.unfreeze()))


class AtariiIQNet:
    def __init__(self, n_nets: int, n_actions: int) -> None:
        self.n_nets = n_nets
        self.iqn_net = AtariIQNNet(n_actions)

    def init(
        self, key_init: jax.random.PRNGKey, state: jnp.ndarray, key: jax.random.PRNGKey, n_quantiles: int
    ) -> FrozenDict:
        return jax.vmap(self.iqn_net.init, in_axes=(0, None, None, None))(
            jax.random.split(key_init, self.n_nets), state, key, n_quantiles
        )

    def apply(
        self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKey, n_quantiles: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # out_axes = 1 because the output shape is (batch_size, n_nets, n_quantiles, n_actions) | (batch_size, n_nets, n_quantiles)
        output, quantiles = jax.vmap(self.iqn_net.apply, in_axes=(0, None, None, None), out_axes=(1, 0))(
            params, state, key, n_quantiles
        )

        # We use the same key for the quantiles here so that the 'apply' function computes the same quantiles over all the heads.
        return output, quantiles[0]

    def apply_other_heads(
        self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKey, n_quantiles: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        params_other_heads = jax.tree_util.tree_map(lambda param: param[1:], params)

        return self.apply(params_other_heads, state, key, n_quantiles)

    def apply_specific_head(
        self,
        params: FrozenDict,
        idx_head: int,
        state: jnp.ndarray,
        key: jax.random.PRNGKey,
        n_quantiles: int,
    ) -> jnp.ndarray:
        params_heads = jax.tree_util.tree_map(lambda param: param[idx_head], params)

        return self.iqn_net.apply(params_heads, state, key, n_quantiles)[0]

    def rolling_step(self, params: FrozenDict) -> FrozenDict:
        return FrozenDict(jax.tree_util.tree_map(lambda param: roll(param), params.unfreeze()))


class AtariiIQN(iIQN):
    def __init__(
        self,
        n_heads: int,
        state_shape: list,
        n_actions: int,
        cumulative_gamma: float,
        network_key: jax.random.PRNGKeyArray,
        head_behaviorial_probability: jnp.ndarray,
        learning_rate: float,
        epsilon_optimizer: float,
        n_training_steps_per_online_update: int,
        n_training_steps_per_target_update: int,
        n_training_steps_per_rolling_step: int,
        n_quantiles_policy: int,
        n_quantiles: int,
        n_quantiles_target: int,
        shared_network: int,
    ) -> None:
        super().__init__(
            n_heads,
            state_shape,
            n_actions,
            cumulative_gamma,
            AtariSharediIQNet(n_heads, n_actions) if shared_network else AtariiIQNet(n_heads, n_actions),
            network_key,
            head_behaviorial_probability,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
            n_training_steps_per_rolling_step,
            n_quantiles_policy,
            n_quantiles,
            n_quantiles_target,
        )
