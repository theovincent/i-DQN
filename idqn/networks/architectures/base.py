import flax.linen as nn
import jax
import jax.numpy as jnp


def scale(state: jnp.ndarray) -> jnp.ndarray:
    return state / 255.0


class Torso(nn.Module):
    initialization_type: str

    @nn.compact
    def __call__(self, state):
        if self.initialization_type == "dqn":
            initializer = nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="truncated_normal")
        elif self.initialization_type == "iqn":
            initializer = nn.initializers.variance_scaling(
                scale=1.0 / jnp.sqrt(3.0), mode="fan_in", distribution="uniform"
            )
        elif self.initialization_type == "rem":
            initializer = nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform")

        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), kernel_init=initializer)(state)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), kernel_init=initializer)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), kernel_init=initializer)(x)
        x = nn.relu(x)

        return x.flatten()


class Head(nn.Module):
    n_actions: int
    initialization_type: str

    @nn.compact
    def __call__(self, x):
        if self.initialization_type == "dqn":
            initializer = nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="truncated_normal")
        elif self.initialization_type == "iqn":
            initializer = nn.initializers.variance_scaling(
                scale=1.0 / jnp.sqrt(3.0), mode="fan_in", distribution="uniform"
            )
        elif self.initialization_type == "rem":
            initializer = nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform")

        x = nn.Dense(features=512, kernel_init=initializer)(x)
        x = nn.relu(x)

        return nn.Dense(features=self.n_actions, kernel_init=initializer)(x)


class QuantileEmbedding(nn.Module):
    n_features: int = 7744
    quantile_embedding_dim: int = 64

    @nn.compact
    def __call__(self, key, n_quantiles):
        initializer = nn.initializers.variance_scaling(scale=1.0 / jnp.sqrt(3.0), mode="fan_in", distribution="uniform")

        quantiles = jax.random.uniform(key, shape=(n_quantiles, 1))
        arange = jnp.arange(1, self.quantile_embedding_dim + 1).reshape((1, self.quantile_embedding_dim))

        quantile_embedding = nn.Dense(features=self.n_features, kernel_init=initializer)(
            jnp.cos(jnp.pi * quantiles @ arange)
        )
        # output (n_quantiles, n_features) | (n_quantiles)
        return (nn.relu(quantile_embedding), jnp.squeeze(quantiles, axis=1))


def roll(param):
    param = param.at[:-1].set(param[1:])

    return param
