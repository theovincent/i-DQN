import haiku as hk
import jax
import jax.numpy as jnp

from idqn.networks.base_q import BaseQ


class FullyConnectedQNet(hk.Module):
    def __init__(self, layers_dimension: list, zero_initializer: bool, n_actions: int) -> None:
        super().__init__(name="FullyConnectedNet")
        self.layers_dimension = layers_dimension
        if zero_initializer:
            self.initializer = hk.initializers.Constant(0)
        else:
            self.initializer = hk.initializers.TruncatedNormal()
        self.n_actions = n_actions

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        x = state

        for idx, layer_dimension in enumerate(self.layers_dimension, start=1):
            x = hk.Linear(layer_dimension, name=f"linear_{idx}")(x)
            x = jax.nn.relu(x)

        x = hk.Linear(self.n_actions, w_init=self.initializer, name="linear_last")(x)

        return x


class FullyConnectedQ(BaseQ):
    def __init__(
        self,
        state_shape: list,
        n_actions: int,
        gamma: float,
        network_key: jax.random.PRNGKeyArray,
        layers_dimension: list,
        zero_initializer: bool,
        learning_rate: dict = None,
    ) -> None:
        def network(state: jnp.ndarray) -> jnp.ndarray:
            return FullyConnectedQNet(layers_dimension, zero_initializer, n_actions)(state)

        super().__init__(
            state_shape=state_shape,
            gamma=gamma,
            network=network,
            network_key=network_key,
            learning_rate=learning_rate,
        )
