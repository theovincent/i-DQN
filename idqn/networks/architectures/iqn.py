from typing import Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp

from idqn.networks.iqn import IQN
from idqn.networks.architectures.base import Torso, Head, QuantileEmbedding


class AtariIQNNet(nn.Module):
    n_actions: int

    def setup(self):
        self.torso = Torso(dqn_initialisation=False)
        self.quantile_embedding = QuantileEmbedding()
        self.head = Head(self.n_actions, dqn_initialisation=False)

    def __call__(self, state, key, n_quantiles) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # output (batch_size, n_features)
        state_features = self.torso(state)
        # output (batch_size, n_quantiles, n_features)
        quantiles_features, quantiles = self.quantile_embedding(key, n_quantiles, state_features.shape[0])

        # mapping over the quantiles. output (batch_size, n_quantiles, n_features)
        multiplied_features = jax.vmap(
            lambda quantile_features, state_features_: quantile_features * state_features_, (1, None), 1
        )(quantiles_features, state_features)

        return (
            self.head(multiplied_features),
            quantiles,
        )  # output (batch_size, n_quantiles, n_actions) | (batch_size, n_quantiles)


class AtariIQN(IQN):
    def __init__(
        self,
        state_shape: list,
        n_actions: int,
        cumulative_gamma: float,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
        epsilon_optimizer: float,
        n_training_steps_per_online_update: int,
        n_training_steps_per_target_update: int,
    ) -> None:
        super().__init__(
            state_shape,
            n_actions,
            cumulative_gamma,
            AtariIQNNet(n_actions),
            network_key,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
        )
