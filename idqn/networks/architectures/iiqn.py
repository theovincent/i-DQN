from typing import Tuple
from flax.core import FrozenDict
import jax
import jax.numpy as jnp

from idqn.networks.iiqn import iIQN
from idqn.networks.architectures.base import Torso, Head, QuantileEmbedding, roll
from idqn.networks.architectures.iqn import AtariIQNNet


class AtariSharediIQNet:
    def __init__(self, n_heads: int, n_actions: int, initialization_type: str) -> None:
        self.n_heads = n_heads
        self.n_actions = n_actions
        self.torso = Torso(initialization_type)
        self.quantile_embedding = QuantileEmbedding()
        self.head = Head(self.n_actions, initialization_type)

    def init(
        self, key_init: jax.random.PRNGKey, state: jnp.ndarray, key: jax.random.PRNGKey, n_quantiles: int
    ) -> FrozenDict:
        # We need two sets of torso parameters
        torso_params = jax.vmap(self.torso.init, in_axes=(0, None))(jax.random.split(key_init), state)

        # We need two set of quantile embedding parameters
        key_init, _ = jax.random.split(key_init)
        quantiles_params = jax.vmap(self.quantile_embedding.init, in_axes=(0, None, None))(
            jax.random.split(key_init), key, n_quantiles
        )

        # Compute the input for the heads
        # output (n_features)
        state_features = self.torso.apply(jax.tree_util.tree_map(lambda param: param[0], torso_params), state)
        # output (n_quantiles, n_features)
        quantiles_features, _ = self.quantile_embedding.apply(
            jax.tree_util.tree_map(lambda param: param[0], quantiles_params), key, n_quantiles
        )

        # mapping over the quantiles | output (n_quantiles, n_features)
        features = jax.vmap(
            lambda quantile_features, state_features_: quantile_features * state_features_,
            in_axes=(0, None),
        )(quantiles_features, state_features)

        head_params = jax.vmap(self.head.init, in_axes=(0, None))(jax.random.split(key_init, self.n_heads), features)

        return FrozenDict(torso_params=torso_params, quantiles_params=quantiles_params, head_params=head_params)

    def apply(
        self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKey, n_quantiles: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # output (2, n_features)
        state_features = jax.vmap(self.torso.apply, in_axes=(0, None))(params["torso_params"], state)

        # We use the same key for the quantiles here so that the 'apply' function computes the same quantiles over all the heads.
        # output (2, n_quantiles, n_features) | (2, n_quantiles)
        quantiles_features, quantiles = jax.vmap(self.quantile_embedding.apply, in_axes=(0, None, None))(
            params["quantiles_params"], key, n_quantiles
        )

        # mapping over the quantiles | output (2, n_quantiles, n_features)
        features = jax.vmap(
            lambda quantile_features_, state_features_: quantile_features_ * state_features_, (1, None), 1
        )(quantiles_features, state_features)

        # output (n_heads, n_quantiles, n_actions)
        repeated_features_ = jnp.repeat(features[None, 1], self.n_heads, axis=0)
        repeated_features = repeated_features_.at[0].set(features[0])

        # output (n_heads, n_quantiles, n_actions) | (n_quantiles)
        return jax.vmap(self.head.apply)(params["head_params"], repeated_features), quantiles[0]

    def apply_other_heads(
        self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKey, n_quantiles: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # output (n_features)
        state_feature = self.torso.apply(jax.tree_util.tree_map(lambda param: param[1], params["torso_params"]), state)

        # output (n_quantiles, n_features) | (n_quantiles)
        quantiles_feature, quantiles = self.quantile_embedding.apply(
            jax.tree_util.tree_map(lambda param: param[1], params["quantiles_params"]), key, n_quantiles
        )

        # mapping over the quantiles | output (n_quantiles, n_features)
        feature = jax.vmap(
            lambda quantile_feature_, state_feature_: quantile_feature_ * state_feature_, in_axes=(0, None)
        )(quantiles_feature, state_feature)

        # output (n_heads, n_quantiles, n_actions)
        output = jax.vmap(self.head.apply, in_axes=(0, None))(
            jax.tree_util.tree_map(lambda param: param[1:], params["head_params"]), feature
        )

        return output, quantiles  # output (n_heads, n_quantiles, n_actions) | (n_quantiles)

    def apply_specific_head(
        self, params: FrozenDict, idx_head: int, state: jnp.ndarray, key: jax.random.PRNGKey, n_quantiles: int
    ) -> jnp.ndarray:
        # output (n_features)
        state_feature = self.torso.apply(
            jax.tree_util.tree_map(
                lambda param: param[jax.lax.cond(idx_head >= 1, lambda: 1, lambda: 0)], params["torso_params"]
            ),
            state,
        )

        # output (n_quantiles, n_features)
        quantiles_feature, _ = self.quantile_embedding.apply(
            jax.tree_util.tree_map(
                lambda param: param[jax.lax.cond(idx_head >= 1, lambda: 1, lambda: 0)], params["quantiles_params"]
            ),
            key,
            n_quantiles,
        )

        # mapping over the quantiles | output (n_quantiles, n_features)
        feature = jax.vmap(
            lambda quantile_feature_, state_feature_: quantile_feature_ * state_feature_, in_axes=(0, None)
        )(quantiles_feature, state_feature)

        return self.head.apply(
            jax.tree_util.tree_map(lambda param: param[idx_head], params["head_params"]), feature
        )  # output (n_quantiles, n_actions)

    def rolling_step(self, params: FrozenDict) -> FrozenDict:
        return FrozenDict(jax.tree_util.tree_map(lambda param: roll(param), params.unfreeze()))


class AtariiIQNet:
    def __init__(self, n_nets: int, n_actions: int, initialization_type: str) -> None:
        self.n_nets = n_nets
        self.iqn_net = AtariIQNNet(n_actions, initialization_type)

    def init(
        self, key_init: jax.random.PRNGKey, state: jnp.ndarray, key: jax.random.PRNGKey, n_quantiles: int
    ) -> FrozenDict:
        return jax.vmap(self.iqn_net.init, in_axes=(0, None, None, None))(
            jax.random.split(key_init, self.n_nets), state, key, n_quantiles
        )

    def apply(
        self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKey, n_quantiles: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # output (n_nets, n_quantiles, n_actions) | (n_nets, n_quantiles)
        output, quantiles = jax.vmap(self.iqn_net.apply, in_axes=(0, None, None, None))(params, state, key, n_quantiles)

        # We use the same key for the quantiles here so that the 'apply' function computes the same quantiles over all the heads.
        return output, quantiles[0]

    def apply_other_heads(
        self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKey, n_quantiles: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        params_other_heads = jax.tree_util.tree_map(lambda param: param[1:], params)

        # output (n_nets - 1, n_quantiles, n_actions) | (n_nets - 1, n_quantiles)
        return self.apply(params_other_heads, state, key, n_quantiles)

    def apply_specific_head(
        self,
        params: FrozenDict,
        idx_head: int,
        state: jnp.ndarray,
        key: jax.random.PRNGKey,
        n_quantiles: int,
    ) -> jnp.ndarray:
        params_head = jax.tree_util.tree_map(lambda param: param[idx_head], params)

        # output (n_quantiles, n_actions)
        return self.iqn_net.apply(params_head, state, key, n_quantiles)[0]

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
            AtariSharediIQNet(n_heads, n_actions, "iqn") if shared_network else AtariiIQNet(n_heads, n_actions, "iqn"),
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
