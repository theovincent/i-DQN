import haiku as hk
import jax
import jax.numpy as jnp

from idqn.networks.base_q import iQ
from idqn.utils.params import add_noise


class FullyConnectedMultiQNet(hk.Module):
    def __init__(
        self,
        n_heads: int,
        shared_layers_dimension: list,
        layers_dimension: list,
        zero_initializer: bool,
        n_actions: int,
    ) -> None:
        self.n_heads = n_heads
        super().__init__(name="FullyConnectedNet")
        if zero_initializer:
            self.initializer = hk.initializers.Constant(0)
        else:
            self.initializer = hk.initializers.TruncatedNormal()
        self.n_actions = n_actions

        shared_layers_first_head_ = []
        shared_layers_other_heads_ = []
        for idx, shared_layer_dimension in enumerate(shared_layers_dimension):
            shared_layers_first_head_.extend(
                [hk.Linear(shared_layer_dimension, name=f"shared_first_head_linear_{idx}"), jax.nn.relu]
            )
            shared_layers_other_heads_.extend(
                [hk.Linear(shared_layer_dimension, name=f"shared_other_heads_linear_{idx}"), jax.nn.relu]
            )
        self.shared_layers_first_head = hk.Sequential(shared_layers_first_head_, name="shared_layers_first_head")
        self.shared_layers_other_heads = hk.Sequential(shared_layers_other_heads_, name="shared_layers")

        self.heads = []
        for idx_head in range(self.n_heads):
            head_ = []
            for idx_layer, layer_dimension in enumerate(layers_dimension):
                head_.extend([hk.Linear(layer_dimension, name=f"head_{idx_head}_linear_{idx_layer}"), jax.nn.relu])
            head_.append(hk.Linear(self.n_actions, w_init=self.initializer, name=f"head_{idx_head}_linear_last"))
            self.heads.append(hk.Sequential(head_, name=f"head_{idx_head}"))

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        input = state
        input = jnp.atleast_2d(input)
        output = jnp.zeros((input.shape[0], self.n_heads, self.n_actions))

        shared_input_first_head = self.shared_layers_first_head(input)
        output = output.at[:, 0].set(self.heads[0](shared_input_first_head))

        shared_input_other_heads = self.shared_layers_other_heads(input)
        for idx_head in range(1, self.n_heads):
            output = output.at[:, idx_head].set(self.heads[idx_head](shared_input_other_heads))

        return output


class FullyConnectedMultiQ(iQ):
    def __init__(
        self,
        importance_iteration: jnp.ndarray,
        state_shape: list,
        n_actions: int,
        gamma: float,
        network_key: jax.random.PRNGKeyArray,
        shared_layers_dimension: list,
        layers_dimension: list,
        zero_initializer: bool,
        learning_rate: float,
    ) -> None:
        self.n_shared_layers = len(shared_layers_dimension)
        self.n_layers_head = len(layers_dimension)

        def network(state: jnp.ndarray) -> jnp.ndarray:
            return FullyConnectedMultiQNet(
                len(importance_iteration) + 1, shared_layers_dimension, layers_dimension, zero_initializer, n_actions
            )(state)

        super().__init__(
            importance_iteration=importance_iteration,
            state_shape=state_shape,
            gamma=gamma,
            network=network,
            network_key=network_key,
            learning_rate=learning_rate,
        )

    def move_forward(self, params: hk.Params) -> hk.Params:
        # Get random params
        randon_params = self.random_init_params()
        for idx in range(self.n_shared_layers):
            randon_params[f"FullyConnectedNet/~/shared_first_head_linear_{idx}"] = {"w": 0, "b": 0}
        for idx in range(self.n_layers_head):
            randon_params[f"FullyConnectedNet/~/head_0_linear_{idx}"] = {"w": 0, "b": 0}
        randon_params[f"FullyConnectedNet/~/head_0_linear_last"] = {"w": 0, "b": 0}

        # The shared params of the first head takes the shared params of the other heads
        for idx in range(self.n_shared_layers):
            params[f"FullyConnectedNet/~/shared_first_head_linear_{idx}"] = add_noise(
                params[f"FullyConnectedNet/~/shared_other_heads_linear_{idx}"],
                randon_params[f"FullyConnectedNet/~/shared_first_head_linear_{idx}"],
            )

        # Each head takes the params of the last head with some noise
        for idx_head in range(self.n_heads):
            for idx_layer in range(self.n_layers_head):
                params[f"FullyConnectedNet/~/head_{idx_head}_linear_{idx_layer}"] = add_noise(
                    params[f"FullyConnectedNet/~/head_{self.n_heads - 1}_linear_{idx_layer}"],
                    randon_params[f"FullyConnectedNet/~/head_{idx_head}_linear_{idx_layer}"],
                )
            params[f"FullyConnectedNet/~/head_{idx_head}_linear_last"] = add_noise(
                params[f"FullyConnectedNet/~/head_{self.n_heads - 1}_linear_last"],
                randon_params[f"FullyConnectedNet/~/head_{idx_head}_linear_last"],
            )

        return params
