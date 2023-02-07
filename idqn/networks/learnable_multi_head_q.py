import haiku as hk
import jax
import jax.numpy as jnp

from idqn.networks.base_q import iQ
from idqn.utils.params import add_noise


class FullyConnectedMultiQNet(hk.Module):
    def __init__(self, n_heads: int, layers_dimension: list, zero_initializer: bool, n_actions: int) -> None:
        super().__init__(name="FullyConnectedNet")
        self.n_heads = n_heads
        self.layers_dimension = layers_dimension
        if zero_initializer:
            self.initializer = hk.initializers.Constant(0)
        else:
            self.initializer = hk.initializers.TruncatedNormal()
        self.n_actions = n_actions

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        input = state
        input = jnp.atleast_2d(input)
        output = jnp.zeros((input.shape[0], self.n_heads, self.n_actions))

        for idx_head in range(self.n_heads):
            for idx, layer_dimension in enumerate(self.layers_dimension):
                x_head = hk.Linear(layer_dimension, name=f"head_{idx_head}_linear_{idx}")(input)
                x_head = jax.nn.relu(x_head)

            output = output.at[:, idx_head].set(
                hk.Linear(self.n_actions, w_init=self.initializer, name=f"head_{idx_head}_linear_last")(x_head)
            )

        return output


class FullyConnectedMultiQ(iQ):
    def __init__(
        self,
        n_heads: int,
        state_shape: list,
        n_actions: int,
        gamma: float,
        network_key: jax.random.PRNGKeyArray,
        layers_dimension: list,
        zero_initializer: bool,
        learning_rate: dict = None,
    ) -> None:
        self.n_layers = len(layers_dimension)

        def network(state: jnp.ndarray) -> jnp.ndarray:
            return FullyConnectedMultiQNet(n_heads, layers_dimension, zero_initializer, n_actions)(state)

        super().__init__(
            n_heads,
            state_shape=state_shape,
            gamma=gamma,
            network=network,
            network_key=network_key,
            learning_rate=learning_rate,
        )

    def move_forward(self, params: hk.Params) -> hk.Params:
        randon_params = self.random_init_params()
        for idx_layer in range(self.n_layers):
            randon_params[f"FullyConnectedNet/head_{0}_linear_{idx_layer}"] = {"w": 0, "b": 0}
        randon_params[f"FullyConnectedNet/head_{0}_linear_last"] = {"w": 0, "b": 0}

        for idx_head in range(self.n_heads):
            for idx_layer in range(self.n_layers):
                params[f"FullyConnectedNet/head_{idx_head}_linear_{idx_layer}"] = add_noise(
                    params[f"FullyConnectedNet/head_{self.n_heads - 1}_linear_{idx_layer}"],
                    randon_params[f"FullyConnectedNet/head_{idx_head}_linear_{idx_layer}"],
                )
            params[f"FullyConnectedNet/head_{idx_head}_linear_last"] = add_noise(
                params[f"FullyConnectedNet/head_{self.n_heads - 1}_linear_last"],
                randon_params[f"FullyConnectedNet/head_{idx_head}_linear_last"],
            )

        return params
