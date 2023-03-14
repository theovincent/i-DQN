from functools import partial
import haiku as hk
import jax
import jax.numpy as jnp

from idqn.networks.base_q import DQN, iDQN
from idqn.utils.params import set_params


class AtariQNet(hk.Module):
    def __init__(self, zero_initializer: bool, n_actions: int) -> None:
        super().__init__(name="AtariQNet")
        if zero_initializer:
            self.initializer = hk.initializers.Constant(0)
        else:
            self.initializer = hk.initializers.TruncatedNormal()
        self.n_actions = n_actions

        self.network = hk.Sequential(
            [
                hk.Conv2D(output_channels=32, kernel_shape=[8, 8], stride=4),
                hk.Conv2D(output_channels=64, kernel_shape=[4, 4], stride=2),
                hk.Conv2D(output_channels=64, kernel_shape=[3, 3], stride=1),
                hk.Flatten(),
                hk.Linear(output_size=512),
                hk.Linear(output_size=n_actions, w_init=self.initializer),
            ],
            name="network",
        )

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        input = state / 255.0
        # "jnp.atleast4d"
        input = jnp.array(input, copy=False, ndmin=4)

        return self.network(input)


class AtariDQN(DQN):
    def __init__(
        self,
        state_shape: list,
        n_actions: int,
        gamma: float,
        network_key: jax.random.PRNGKeyArray,
        zero_initializer: bool,
        learning_rate: float,
    ) -> None:
        self.n_layers = 5

        def network(state: jnp.ndarray) -> jnp.ndarray:
            return AtariQNet(zero_initializer, n_actions)(state)

        super().__init__(state_shape, n_actions, gamma, network, network_key, learning_rate)


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
        super().__init__(name="FullyConnectedMultiQNet")
        if zero_initializer:
            initializer = hk.initializers.Constant(0)
        else:
            initializer = hk.initializers.TruncatedNormal()
        self.n_actions = n_actions

        shared_layers_first_head_ = []
        shared_layers_other_heads_ = []
        for idx_layer, shared_layer_dimension in enumerate(shared_layers_dimension):
            shared_layers_first_head_.extend(
                [hk.Linear(shared_layer_dimension, name=f"shared_first_head_linear_{idx_layer}"), jax.nn.relu]
            )
            shared_layers_other_heads_.extend(
                [hk.Linear(shared_layer_dimension, name=f"shared_other_heads_linear_{idx_layer}"), jax.nn.relu]
            )
        self.shared_layers_first_head = hk.Sequential(shared_layers_first_head_, name="shared_layers_first_head")
        self.shared_layers_other_heads = hk.Sequential(shared_layers_other_heads_, name="shared_layers")

        self.heads = []
        for idx_head in range(self.n_heads):
            head_ = []
            for idx_layer, layer_dimension in enumerate(layers_dimension):
                head_.extend([hk.Linear(layer_dimension, name=f"head_{idx_head}_linear_{idx_layer}"), jax.nn.relu])
            head_.append(hk.Linear(self.n_actions, w_init=initializer, name=f"head_{idx_head}_linear_last"))
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


class FullyConnectediDQN(iDQN):
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
            importance_iteration, state_shape, n_actions, network, gamma, network, network_key, learning_rate
        )

    # This should go to base_q
    def move_forward(self, params: hk.Params) -> hk.Params:
        raise NotImplementedError
        # # Get random params
        # random_params = self.random_init_params()
        # for idx_layer in range(self.n_shared_layers):
        #     random_params[f"FullyConnectedNet/~/shared_first_head_linear_{idx_layer}"] = {"w": 0, "b": 0}
        # for idx_layer in range(self.n_layers_head):
        #     random_params[f"FullyConnectedNet/~/head_0_linear_{idx_layer}"] = {"w": 0, "b": 0}
        # random_params[f"FullyConnectedNet/~/head_0_linear_last"] = {"w": 0, "b": 0}

        # # The shared params of the first head takes the shared params of the other heads
        # for idx_layer in range(self.n_shared_layers):
        #     params[f"FullyConnectedNet/~/shared_first_head_linear_{idx_layer}"] = add_noise(
        #         params[f"FullyConnectedNet/~/shared_other_heads_linear_{idx_layer}"],
        #         random_params[f"FullyConnectedNet/~/shared_first_head_linear_{idx_layer}"],
        #     )

        # # Each head takes the params of the last head with some noise
        # for idx_head in range(self.n_heads):
        #     for idx_layer in range(self.n_layers_head):
        #         params[f"FullyConnectedNet/~/head_{idx_head}_linear_{idx_layer}"] = add_noise(
        #             params[f"FullyConnectedNet/~/head_{self.n_heads - 1}_linear_{idx_layer}"],
        #             random_params[f"FullyConnectedNet/~/head_{idx_head}_linear_{idx_layer}"],
        #         )
        #     params[f"FullyConnectedNet/~/head_{idx_head}_linear_last"] = add_noise(
        #         params[f"FullyConnectedNet/~/head_{self.n_heads - 1}_linear_last"],
        #         random_params[f"FullyConnectedNet/~/head_{idx_head}_linear_last"],
        #     )

        # return params


class AtariMultiQNet(hk.Module):
    def __init__(
        self,
        n_heads: int,
        n_shared_layers: int,
        zero_initializer: bool,
        n_actions: int,
    ) -> None:
        self.n_heads = n_heads
        super().__init__(name="AtariMultiQNet")
        if zero_initializer:
            self.initializer = hk.initializers.Constant(0)
        else:
            self.initializer = hk.initializers.TruncatedNormal()
        self.n_actions = n_actions

        architecture = [
            (hk.Conv2D, {"output_channels": 32, "kernel_shape": [8, 8], "stride": 4}),
            (hk.Conv2D, {"output_channels": 64, "kernel_shape": [4, 4], "stride": 2}),
            (hk.Conv2D, {"output_channels": 64, "kernel_shape": [3, 3], "stride": 1}),
            (hk.Linear, {"output_size": 512}),
            (hk.Linear, {"output_size": n_actions, "w_init": self.initializer}),
        ]

        shared_layers_first_head_ = []
        shared_layers_other_heads_ = []
        for idx_layer in range(n_shared_layers):
            shared_layers_first_head_.extend(
                [
                    architecture[idx_layer][0](
                        **architecture[idx_layer][1], name=f"shared_first_head_layer_{idx_layer}"
                    ),
                    jax.nn.relu,
                ]
            )
            shared_layers_other_heads_.extend(
                [
                    architecture[idx_layer][0](
                        **architecture[idx_layer][1], name=f"shared_other_heads_layer_{idx_layer}"
                    ),
                    jax.nn.relu,
                ]
            )
            if idx_layer == 2:
                shared_layers_first_head_.append(hk.Flatten())
                shared_layers_other_heads_.append(hk.Flatten())

        self.shared_layers_first_head = hk.Sequential(shared_layers_first_head_, name="shared_layers_first_head")
        self.shared_layers_other_heads = hk.Sequential(shared_layers_other_heads_, name="shared_layers_other_heads")

        self.heads = []
        for idx_head in range(self.n_heads):
            head_ = []
            for idx_layer in range(n_shared_layers, len(architecture) - 1):
                head_.extend(
                    [
                        architecture[idx_layer][0](
                            **architecture[idx_layer][1], name=f"head_{idx_head}_layer_{idx_layer}"
                        ),
                        jax.nn.relu,
                    ]
                )
                if idx_layer == 2:
                    head_.append(hk.Flatten())
            head_.append(
                architecture[-1][0](**architecture[-1][1], name=f"head_{idx_head}_layer_{len(architecture) - 1}")
            )
            self.heads.append(hk.Sequential(head_, name=f"head_{idx_head}"))

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        input = state / 255.0
        # "jnp.atleast4d"
        input = jnp.array(input, copy=False, ndmin=4)
        output = jnp.zeros((input.shape[0], self.n_heads, self.n_actions))

        shared_input_first_head = self.shared_layers_first_head(input)
        output = output.at[:, 0].set(self.heads[0](shared_input_first_head))

        shared_input_other_heads = self.shared_layers_other_heads(input)
        for idx_head in range(1, self.n_heads):
            output = output.at[:, idx_head].set(self.heads[idx_head](shared_input_other_heads))

        return output


class AtariiDQN(iDQN):
    def __init__(
        self,
        importance_iteration: jnp.ndarray,
        state_shape: list,
        n_actions: int,
        gamma: float,
        network_key: jax.random.PRNGKeyArray,
        n_shared_layers: int,
        zero_initializer: bool,
        learning_rate: float,
    ) -> None:
        self.n_shared_layers = n_shared_layers
        self.n_layers = 5

        def network(state: jnp.ndarray) -> jnp.ndarray:
            return AtariMultiQNet(len(importance_iteration) + 1, n_shared_layers, zero_initializer, n_actions)(state)

        super().__init__(importance_iteration, state_shape, n_actions, gamma, network, network_key, learning_rate)

    # This should go to base_q
    @partial(jax.jit, static_argnames="self")
    def move_forward(self, params: hk.Params) -> hk.Params:
        raise NotImplementedError
        # # The shared params of the first head takes the shared params of the other heads
        # for idx_layer in range(self.n_shared_layers):
        #     set_params(
        #         params[f"AtariNet/~/shared_first_head_layer_{idx_layer}"],
        #         params[f"AtariNet/~/shared_other_heads_layer_{idx_layer}"],
        #     )

        # # Each head takes the params of the last head with some noise
        # for idx_head in range(self.n_heads):
        #     for idx_layer in range(self.n_shared_layers, self.n_layers):
        #         set_params(
        #             params[f"AtariNet/~/head_{idx_head}_layer_{idx_layer}"],
        #             params[f"AtariNet/~/head_{self.n_heads - 1}_layer_{idx_layer}"],
        #         )

        # return params
