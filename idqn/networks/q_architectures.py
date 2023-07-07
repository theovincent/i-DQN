from functools import partial
import flax.linen as nn
from flax.core import FrozenDict
import jax
import jax.numpy as jnp

from idqn.networks.base_q import DQN, iDQN


class Torso(nn.Module):
    dqn_initialisation: bool = True

    @nn.compact
    def __call__(self, state):
        if self.dqn_initialisation:
            initializer = nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="truncated_normal")
        else:
            initializer = nn.initializers.variance_scaling(
                scale=1.0 / jnp.sqrt(3.0), mode="fan_in", distribution="uniform"
            )

        # scale -> at least 4 dimensions -> transpose to channel last
        preprocessed_state = jnp.array(state / 255.0, ndmin=4).transpose((0, 2, 3, 1))
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), kernel_init=initializer)(preprocessed_state)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), kernel_init=initializer)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), kernel_init=initializer)(x)
        x = nn.relu(x)

        return x.reshape((preprocessed_state.shape[0], -1))  # flatten


class Head(nn.Module):
    n_actions: int
    dqn_initialisation: bool = True

    @nn.compact
    def __call__(self, x):
        if self.dqn_initialisation:
            initializer = nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="truncated_normal")
        else:
            initializer = nn.initializers.variance_scaling(
                scale=1.0 / jnp.sqrt(3.0), mode="fan_in", distribution="uniform"
            )

        x = nn.Dense(features=512, kernel_init=initializer)(x)
        x = nn.relu(x)

        return nn.Dense(features=self.n_actions, kernel_init=initializer)(x)


class QuantileEmbedding(nn.Module):
    n_features: int = 7744
    quantile_embedding_dim: int = 64

    @nn.compact
    def __call__(self, key, n_quantiles, batch_size):
        initializer = nn.initializers.variance_scaling(scale=1.0 / jnp.sqrt(3.0), mode="fan_in", distribution="uniform")

        quantiles = jax.random.uniform(key, shape=(batch_size, n_quantiles, 1))
        arange = jnp.arange(1, self.quantile_embedding_dim + 1).reshape((1, self.quantile_embedding_dim))

        quantile_embedding = nn.Dense(features=self.n_features, kernel_init=initializer)(
            jnp.cos(jnp.pi * quantiles @ arange)
        )
        return nn.relu(quantile_embedding)


class AtariQNet(nn.Module):
    n_actions: int

    def setup(self):
        self.torso = Torso()
        self.head = Head(self.n_actions)

    @nn.compact
    def __call__(self, state):
        return self.head(self.torso(state))


class AtariDQN(DQN):
    def __init__(
        self,
        state_shape: list,
        n_actions: int,
        gamma: float,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
        n_training_steps_per_online_update: int,
        n_training_steps_per_target_update: int,
    ) -> None:
        super().__init__(
            state_shape,
            n_actions,
            gamma,
            AtariQNet(n_actions),
            network_key,
            learning_rate,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
        )


class AtariIQNNet(nn.Module):
    n_actions: int

    def setup(self):
        self.torso = Torso(dqn_initialisation=False)
        self.quantile_embedding = QuantileEmbedding()
        self.head = Head(self.n_actions, dqn_initialisation=False)

    @nn.compact
    def __call__(self, state, key, n_quantiles):
        states_features = self.torso(state)  # output (batch_size, n_features)
        quantile_features = self.quantile_embedding(
            key, n_quantiles, states_features.shape[0]
        )  # output (batch_size, n_quantiles, n_features)

        return self.head(quantile_features * jnp.repeat(states_features[:, jnp.newaxis], n_quantiles, axis=1))


class AtariIQN(DQN):
    def __init__(
        self,
        state_shape: list,
        n_actions: int,
        gamma: float,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
        n_training_steps_per_online_update: int,
        n_training_steps_per_target_update: int,
    ) -> None:
        super().__init__(
            state_shape,
            n_actions,
            gamma,
            AtariIQNNet(n_actions),
            network_key,
            learning_rate,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
        )


class AtariMultiQNet:
    def __init__(self, n_heads: int, n_actions: int) -> None:
        self.n_heads = n_heads
        self.n_actions = n_actions
        self.atari_q_net = AtariQNet(self.n_actions)

    def init(self, key: jax.random.PRNGKey, state: jnp.ndarray) -> FrozenDict:
        return jax.vmap(self.atari_q_net.init, in_axes=[0, None])(jax.random.split(key, self.n_heads), state)

    def apply(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self.atari_q_net.apply, in_axes=[0, None], out_axes=1)(params, state)


class AtariSharedMultiQNet:
    def __init__(self, n_heads: int, n_actions: int) -> None:
        self.n_heads = n_heads
        self.n_actions = n_actions
        self.torso = Torso()
        self.head = Head(self.n_actions)

    def init(self, key: jax.random.PRNGKey, state: jnp.ndarray) -> FrozenDict:
        # We need only two sets of torso parameters
        torso_params = {}
        torso_params["torso_params_0"] = self.torso.init(key, state)
        key, _ = jax.random.split(key)
        torso_params["torso_params_1"] = self.torso.init(key, state)

        features = self.torso.apply(torso_params["torso_params_0"], state)

        head_params = {}
        for idx_head in range(self.n_heads):
            key, _ = jax.random.split(key)
            head_params[f"head_params_{idx_head}"] = self.head.init(key, features)

        return FrozenDict(**torso_params, **head_params)

    def apply(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        features_0 = self.torso.apply(params["torso_params_0"], state)
        features_1 = self.torso.apply(params["torso_params_1"], state)

        # batch_size = features_0.shape[0]
        output = jnp.zeros((features_0.shape[0], self.n_heads, self.n_actions))

        output = output.at[:, 0].set(self.head.apply(params["head_params_0"], features_0))
        for idx_head in range(1, self.n_heads):
            output = output.at[:, idx_head].set(self.head.apply(params[f"head_params_{idx_head}"], features_1))

        return output

    def apply_specific_head(self, params: FrozenDict, state: jnp.ndarray, idx_head: int) -> jnp.ndarray:
        if idx_head == 0:
            return self.apply_first_head(params, state)
        else:
            return self.apply_other_head(params, params[f"head_params_{idx_head}"], state)

    @partial(jax.jit, static_argnames="self")
    def apply_first_head(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        features_0 = self.torso.apply(params["torso_params_0"], state)

        return self.head.apply(params["head_params_0"], features_0)

    @partial(jax.jit, static_argnames="self")
    def apply_other_head(self, params: FrozenDict, params_head: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        features_1 = self.torso.apply(params["torso_params_1"], state)

        return self.head.apply(params_head, features_1)


class AtariiDQN(iDQN):
    def __init__(
        self,
        importance_iteration: jnp.ndarray,
        state_shape: list,
        n_actions: int,
        gamma: float,
        network_key: jax.random.PRNGKeyArray,
        head_behaviorial_probability: jnp.ndarray,
        learning_rate: float,
        n_training_steps_per_online_update: int,
        n_training_steps_per_target_update: int,
        n_training_steps_per_head_update: int,
    ) -> None:
        super().__init__(
            importance_iteration,
            state_shape,
            n_actions,
            gamma,
            AtariSharedMultiQNet(len(importance_iteration) + 1, n_actions),
            network_key,
            head_behaviorial_probability,
            learning_rate,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
            n_training_steps_per_head_update,
        )

    @partial(jax.jit, static_argnames="self")
    def update_heads(self, params: FrozenDict) -> FrozenDict:
        unfrozen_params = params.unfreeze()
        # The shared params of the first head takes the shared params of the other heads
        unfrozen_params["torso_params_0"] = params["torso_params_1"]

        # Each head takes the params of the following one
        for idx_head in range(self.n_heads - 1):
            unfrozen_params[f"head_params_{idx_head}"] = params[f"head_params_{idx_head + 1}"]

        return FrozenDict(unfrozen_params)
