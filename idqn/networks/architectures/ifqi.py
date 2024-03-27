from functools import partial
from typing import Sequence, Union
from flax.core import FrozenDict
import numpy as np
import jax
import jax.numpy as jnp

from idqn.networks.idqn import iDQN
from idqn.networks.architectures.base import MLP, FeatureNet, PolynomialFeature, SineFeature, TileFeature, roll
from idqn.utils.pickle import load_pickled_data
from idqn.sample_collection import IDX_RB


class CarOnHilliFQINet:
    def __init__(self, n_nets: int, features: Sequence, n_actions: int) -> None:
        self.n_nets = n_nets
        self.fqi_net = MLP(features, n_actions)

    def init(self, key_init: jax.random.PRNGKey, state: jnp.ndarray) -> FrozenDict:
        # Initialize each network at the same location
        # jnp.squeeze() because of stack_size = 1
        return jax.vmap(self.fqi_net.init, in_axes=(0, None))(
            jnp.repeat(key_init[None], self.n_nets, axis=0), jnp.squeeze(state)
        )

    def apply(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        # output (n_nets, n_actions)
        # jnp.squeeze() because of stack_size = 1
        return jax.vmap(self.fqi_net.apply, in_axes=(0, None))(params, jnp.squeeze(state))

    def apply_other_heads(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        params_other_heads = jax.tree_util.tree_map(lambda param: param[1:], params)

        return self.apply(params_other_heads, state)

    def rolling_step(self, params: FrozenDict) -> FrozenDict:
        return FrozenDict(jax.tree_util.tree_map(lambda param: roll(param), params.unfreeze()))


class CarOnHilliFQILinearNet:
    def __init__(self, n_nets: int, features: str, n_actions: int) -> None:
        self.n_nets = n_nets

        self.feature_net = FeatureNet([35])
        self.feature_net.init(jax.random.PRNGKey(0), jnp.array([0, 0]))
        self.feature_net.params = load_pickled_data(features)
        # self.feature_net = PolynomialFeature()
        # self.feature_net.params = 0

        self.fqi_net = MLP([], n_actions, use_bias=False)

    def init(self, key_init: jax.random.PRNGKey, state: jnp.ndarray) -> FrozenDict:
        # Initialize each network at the same location
        return jax.vmap(
            lambda key, state: self.fqi_net.init(
                key, self.feature_net.apply(self.feature_net.params, jnp.squeeze(state))
            ),
            in_axes=(0, None),
        )(jnp.repeat(key_init[None], self.n_nets, axis=0), state)

    def apply(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        # output (n_nets, n_actions)
        # jnp.squeeze() because of stack_size = 1
        return jax.vmap(
            lambda param, state: self.fqi_net.apply(
                param, self.feature_net.apply(self.feature_net.params, jnp.squeeze(state))
            ),
            in_axes=(0, None),
        )(params, jnp.squeeze(state))

    def apply_other_heads(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        params_other_heads = jax.tree_util.tree_map(lambda param: param[1:], params)

        return self.apply(params_other_heads, state)

    def rolling_step(self, params: FrozenDict) -> FrozenDict:
        return FrozenDict(jax.tree_util.tree_map(lambda param: roll(param), params.unfreeze()))


class CarOnHilliFQI(iDQN):
    def __init__(
        self,
        n_heads: int,
        features: Union[Sequence, str],
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
            (
                CarOnHilliFQINet(n_heads, features, n_actions)
                if type(features) == Sequence
                else CarOnHilliFQILinearNet(n_heads, features, n_actions)
            ),
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

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKeyArray) -> jnp.int8:
        return jnp.argmax(self.network.apply(params, state)).astype(jnp.int8)

    @partial(jax.jit, static_argnames="self")
    def compute_proposition_value(self, params, target_params, dataset, states, gamma):
        previous_loss = self.loss_on_batch(target_params, target_params, dataset, None)
        current_loss = self.loss_on_batch(params, target_params, dataset, None)
        # We do not compute the shift for the first and last Q-functions.
        # The first did not move. The last one can move freely.
        if self.n_heads > 2:
            shift = (
                jax.vmap(self.compute_norm_2, in_axes=(None, None, 0), out_axes=1)(
                    jax.tree_map(lambda param: param[1:-1], params),
                    jax.tree_map(lambda param: param[1:-1], target_params),
                    states,
                )
                .max(axis=(1, 2))  # over the state-action pairs
                .mean()  # over the heads
            )
        else:
            shift = 0

        return jnp.around(previous_loss - current_loss - gamma * shift, 4)

    def compute_norm_2(self, params, target_params, state):
        return self.metric(self.apply(params, state) - self.apply(target_params, state), ord="2")

    @partial(jax.jit, static_argnames="self")
    def compute_diff_approximation_errors(self, params, target_params, dataset):
        return jnp.around(
            self.loss_on_batch(target_params, target_params, dataset, None)
            - self.loss_on_batch(params, params, dataset, None),
            4,
        )

    def update_bellman_iteration(self, params, dataset):
        # shape (n_samples, 1)
        targets = jax.vmap(self.compute_target, in_axes=(None, 0))(params, dataset)
        # shape (n_features, n_samples)
        features = jax.vmap(
            lambda sample: self.network.feature_net.apply(
                self.network.feature_net.params, jnp.squeeze(sample[IDX_RB["state"]])
            ),
            out_axes=1,
        )(dataset)

        idx_action_0 = dataset[IDX_RB["action"]] == 0
        targets_action_0 = targets[idx_action_0]
        targets_action_1 = targets[~idx_action_0]
        features_action_0 = features[:, idx_action_0]
        features_action_1 = features[:, ~idx_action_0]

        # shape (n_features)
        params_action_0 = np.linalg.inv(features_action_0 @ features_action_0.T) @ features_action_0 @ targets_action_0
        params_action_1 = np.linalg.inv(features_action_1 @ features_action_1.T) @ features_action_1 @ targets_action_1

        # shape (n_features, 2)
        new_params = jnp.hstack((params_action_0, params_action_1))

        unfrozen_params = self.params.unfreeze()
        # shape (2, n_features, 2)
        unfrozen_params["params"]["Dense_0"]["kernel"] = jnp.repeat(new_params[None], 2, axis=0)
        self.params = FrozenDict(unfrozen_params)
