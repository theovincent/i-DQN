import flax.linen as nn
from flax.core import FrozenDict
import jax
import jax.numpy as jnp

from idqn.networks.base_q import DQN, IQN, iDQN, iIQN


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

        # scale -> at least 4 dimensions
        preprocessed_state = jnp.array(state / 255.0, ndmin=4)
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
        return (
            nn.relu(quantile_embedding),
            quantiles[:, :, 0],
        )  # output (batch_size, n_quantiles, n_features) | (batch_size, n_quantiles)


class AtariDQNNet(nn.Module):
    n_actions: int

    def setup(self):
        self.torso = Torso()
        self.head = Head(self.n_actions)

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
        epsilon_optimizer: float,
        n_training_steps_per_online_update: int,
        n_training_steps_per_target_update: int,
    ) -> None:
        super().__init__(
            state_shape,
            n_actions,
            gamma,
            AtariDQNNet(n_actions),
            network_key,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
        )


class AtariIQNNet(nn.Module):
    n_actions: int

    def setup(self):
        self.torso = Torso(dqn_initialisation=False)
        self.quantile_embedding = QuantileEmbedding()
        self.head = Head(self.n_actions, dqn_initialisation=False)

    def __call__(self, state, key, n_quantiles):
        state_features = self.torso(state)  # output (batch_size, n_features)
        quantiles_features, quantiles = self.quantile_embedding(
            key, n_quantiles, state_features.shape[0]
        )  # output (batch_size, n_quantiles, n_features)

        # mapping over the quantiles
        multiplied_features = jax.vmap(
            lambda quantile_features, state_features_: quantile_features * state_features_, (1, None), 1
        )(
            quantiles_features, state_features
        )  # output (batch_size, n_quantiles, n_features)

        return (
            self.head(multiplied_features),
            quantiles,
        )  # output (batch_size, n_quantiles, n_actions) | (batch_size, n_quantiles)


class AtariIQN(IQN):
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
            AtariIQNNet(n_actions),
            network_key,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
        )


class AtariSharediDQNNet:
    def __init__(self, n_heads: int, n_actions: int) -> None:
        self.n_heads = n_heads
        self.n_actions = n_actions
        self.torso = Torso()
        self.head = Head(self.n_actions)

    def init(self, key_init: jax.random.PRNGKey, state: jnp.ndarray) -> FrozenDict:
        # We need only two sets of torso parameters
        torso_params = {}
        torso_params["torso_params_0"] = self.torso.init(key_init, state)
        key_init, _ = jax.random.split(key_init)
        torso_params["torso_params_1"] = self.torso.init(key_init, state)

        features = self.torso.apply(torso_params["torso_params_0"], state)

        head_params = {}
        for idx_head in range(self.n_heads):
            key_init, _ = jax.random.split(key_init)
            head_params[f"head_params_{idx_head}"] = self.head.init(key_init, features)

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

    def apply_other_heads(self, params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        features_1 = self.torso.apply(params["torso_params_1"], state)

        # batch_size = features_1.shape[0]
        output = jnp.zeros((features_1.shape[0], self.n_heads - 1, self.n_actions))

        for idx_head in range(1, self.n_heads):
            output = output.at[:, idx_head - 1].set(self.head.apply(params[f"head_params_{idx_head}"], features_1))

        return output

    def apply_specific_head(self, torso_params: FrozenDict, head_params: FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        features = self.torso.apply(torso_params, state)

        return self.head.apply(head_params, features)

    def update_heads(self, params: FrozenDict) -> FrozenDict:
        unfrozen_params = params.unfreeze()
        # The shared params of the first head takes the shared params of the other heads
        unfrozen_params["torso_params_0"] = params["torso_params_1"]

        # Each head takes the params of the following one
        for idx_head in range(self.n_heads - 1):
            unfrozen_params[f"head_params_{idx_head}"] = params[f"head_params_{idx_head + 1}"]

        return FrozenDict(unfrozen_params)


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
        n_training_steps_per_head_update: int,
    ) -> None:
        super().__init__(
            n_heads,
            state_shape,
            n_actions,
            gamma,
            AtariSharediDQNNet(n_heads, n_actions),
            network_key,
            head_behaviorial_probability,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
            n_training_steps_per_head_update,
        )


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
        torso_params = {}
        torso_params["torso_params_0"] = self.torso.init(key_init, state)
        key_init, _ = jax.random.split(key_init)
        torso_params["torso_params_1"] = self.torso.init(key_init, state)

        # We need two set of quantile embedding parameters
        quantiles_params = {}
        key_init, _ = jax.random.split(key_init)
        quantiles_params["quantiles_params_0"] = self.quantile_embedding.init(
            key_init, key, n_quantiles, 1
        )  # 1 for batch_size
        key_init, _ = jax.random.split(key_init)
        quantiles_params["quantiles_params_1"] = self.quantile_embedding.init(
            key_init, key, n_quantiles, 1
        )  # 1 for batch_sizes

        # Compute the input for the heads
        state_features = self.torso.apply(torso_params["torso_params_0"], state)  # output (batch_size, n_features)
        quantiles_features, _ = self.quantile_embedding.apply(
            quantiles_params["quantiles_params_0"], key, n_quantiles, 1
        )  # output (batch_size, n_quantiles, n_features)

        # mapping first over the states and then over the quantiles
        multiplied_features = jax.vmap(
            jax.vmap(lambda quantile_features, state_features_: quantile_features * state_features_, (0, None))
        )(
            quantiles_features, state_features
        )  # output (batch_size, n_quantiles, n_features)

        head_params = {}
        for idx_head in range(self.n_heads):
            key_init, _ = jax.random.split(key_init)
            head_params[f"head_params_{idx_head}"] = self.head.init(key_init, multiplied_features)

        return FrozenDict(**torso_params, **quantiles_params, **head_params)

    def apply(self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKey, n_quantiles: int) -> jnp.ndarray:
        state_features_0 = self.torso.apply(params["torso_params_0"], state)  # output (batch_size, n_features)
        state_features_1 = self.torso.apply(params["torso_params_1"], state)  # output (batch_size, n_features)

        # batch_size = features_0.shape[0]
        quantiles_features_0, quantiles = self.quantile_embedding.apply(
            params["quantiles_params_0"], key, n_quantiles, state_features_0.shape[0]
        )  # output (batch_size, n_quantiles, n_features) | (batch_size, n_quantiles)
        # We use the same key for the quantiles here so that the 'apply' function computes the same quantiles over all the heads.
        quantiles_features_1, _ = self.quantile_embedding.apply(
            params["quantiles_params_1"], key, n_quantiles, state_features_1.shape[0]
        )  # output (batch_size, n_quantiles, n_features)

        # mapping over the quantiles
        multiplied_features_0 = jax.vmap(
            lambda quantile_features, state_features_: quantile_features * state_features_, (1, None), 1
        )(
            quantiles_features_0, state_features_0
        )  # output (batch_size, n_quantiles, n_features)
        multiplied_features_1 = jax.vmap(
            lambda quantile_features, state_features_: quantile_features * state_features_, (1, None), 1
        )(
            quantiles_features_1, state_features_1
        )  # output (batch_size, n_quantiles, n_features)

        output = jnp.zeros((state_features_0.shape[0], self.n_heads, n_quantiles, self.n_actions))

        output = output.at[:, 0].set(
            self.head.apply(params["head_params_0"], multiplied_features_0)
        )  # output (batch_size, n_quantiles, n_actions)
        for idx_head in range(1, self.n_heads):
            output = output.at[:, idx_head].set(
                self.head.apply(params[f"head_params_{idx_head}"], multiplied_features_1)
            )  # output (batch_size, n_quantiles, n_actions)

        return output, quantiles  # output (batch_size, n_heads, n_quantiles, n_actions) | (batch_size, n_quantiles)

    def apply_other_heads(
        self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKey, n_quantiles: int
    ) -> jnp.ndarray:
        state_features_1 = self.torso.apply(params["torso_params_1"], state)  # output (batch_size, n_features)

        # batch_size = features_1.shape[0]
        quantiles_features_1, quantiles = self.quantile_embedding.apply(
            params["quantiles_params_1"], key, n_quantiles, state_features_1.shape[0]
        )  # output (batch_size, n_quantiles, n_features) | (batch_size, n_quantiles)

        # mapping over the quantiles
        multiplied_features_1 = jax.vmap(
            lambda quantile_features, state_features_: quantile_features * state_features_, (1, None), 1
        )(
            quantiles_features_1, state_features_1
        )  # output (batch_size, n_quantiles, n_features)

        output = jnp.zeros((state_features_1.shape[0], self.n_heads - 1, n_quantiles, self.n_actions))

        for idx_head in range(1, self.n_heads):
            output = output.at[:, idx_head - 1].set(
                self.head.apply(params[f"head_params_{idx_head}"], multiplied_features_1)
            )  # output (batch_size, n_quantiles, n_actions)

        return output, quantiles  # output (batch_size, n_heads, n_quantiles, n_actions) | (batch_size, n_quantiles)

    def apply_specific_head(
        self,
        torso_params: FrozenDict,
        quantiles_params: FrozenDict,
        head_params: FrozenDict,
        state: jnp.ndarray,
        key: jax.random.PRNGKey,
        n_quantiles: int,
    ) -> jnp.ndarray:
        state_features = self.torso.apply(torso_params, state)  # output (batch_size, n_features)

        # batch_size = features_0.shape[0]
        quantiles_features, _ = self.quantile_embedding.apply(
            quantiles_params, key, n_quantiles, state_features.shape[0]
        )  # output (batch_size, n_quantiles, n_features)

        # mapping over the quantiles
        multiplied_features = jax.vmap(
            lambda quantile_features, state_features: quantile_features * state_features, (1, None), 1
        )(
            quantiles_features, state_features
        )  # output (batch_size, n_quantiles, n_features)

        return self.head.apply(head_params, multiplied_features)  # output (batch_size, n_quantiles, n_actions)

    def update_heads(self, params: FrozenDict) -> FrozenDict:
        unfrozen_params = params.unfreeze()
        # The shared params of the first head takes the shared params of the other heads
        unfrozen_params["torso_params_0"] = params["torso_params_1"]
        unfrozen_params["quantiles_params_0"] = params["quantiles_params_1"]

        # Each head takes the params of the following one
        for idx_head in range(self.n_heads - 1):
            unfrozen_params[f"head_params_{idx_head}"] = params[f"head_params_{idx_head + 1}"]

        return FrozenDict(unfrozen_params)


class AtariiIQN(iIQN):
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
        n_training_steps_per_head_update: int,
        n_quantiles_policy: int,
        n_quantiles: int,
        n_quantiles_target: int,
    ) -> None:
        super().__init__(
            n_heads,
            state_shape,
            n_actions,
            gamma,
            AtariSharediIQNet(n_heads, n_actions),
            network_key,
            head_behaviorial_probability,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
            n_training_steps_per_head_update,
            n_quantiles_policy,
            n_quantiles,
            n_quantiles_target,
        )
