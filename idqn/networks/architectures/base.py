import flax.linen as nn
import jax
import jax.numpy as jnp


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


def roll(param):
    param = param.at[:-1].set(param[1:])

    return param
