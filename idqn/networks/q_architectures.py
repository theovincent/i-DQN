from functools import partial
import flax.linen as nn
from flax.core import FrozenDict
import jax
import jax.numpy as jnp

from idqn.networks.base_q import DQN, iDQN


class AtariQNet(nn.Module):
    n_actions: int
    initializer = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, state) -> jnp.float32:
        initializer = nn.initializers.xavier_uniform()
        # Convert to channel last
        x = jnp.transpose(state / 255.0, (1, 2, 0))
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), kernel_init=initializer)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), kernel_init=initializer)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), kernel_init=initializer)(x)
        x = nn.relu(x)
        x = x.reshape((-1))  # flatten
        x = nn.Dense(features=512, kernel_init=initializer)(x)
        x = nn.relu(x)

        return nn.Dense(features=self.n_actions, kernel_init=initializer)(x)


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


class AtariSharedMultiQNet(nn.Module):
    n_heads: int
    n_actions: int
    initializer = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, state) -> jnp.float32:
        initializer = nn.initializers.xavier_uniform()
        # Convert to channel last
        input_shared_params = jnp.transpose(state / 255.0, (1, 2, 0))

        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), kernel_init=initializer)(input_shared_params)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), kernel_init=initializer)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), kernel_init=initializer)(x)
        x = nn.relu(x)
        input_first_head = x.reshape((-1))  # flatten

        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), kernel_init=initializer)(input_shared_params)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), kernel_init=initializer)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), kernel_init=initializer)(x)
        x = nn.relu(x)
        input_other_heads = x.reshape((-1))  # flatten

        output = jnp.zeros((self.n_heads, self.n_actions))

        x = nn.Dense(features=512, kernel_init=initializer)(input_first_head)
        x = nn.relu(x)
        output = output.at[0].set(nn.Dense(features=self.n_actions, kernel_init=initializer)(x))

        for idx_head in range(1, self.n_heads):
            x = nn.Dense(features=512, kernel_init=initializer)(input_other_heads)
            x = nn.relu(x)
            output = output.at[idx_head].set(nn.Dense(features=self.n_actions, kernel_init=initializer)(x))

        return output


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
        unfrozen_params["params"]["Conv_0"] = params["params"]["Conv_3"]
        unfrozen_params["params"]["Conv_1"] = params["params"]["Conv_4"]
        unfrozen_params["params"]["Conv_2"] = params["params"]["Conv_5"]

        # Each head takes the params of the last head
        for idx_head in range(self.n_heads - 1):
            unfrozen_params["params"][f"Dense_{2 * idx_head}"] = params["params"][f"Dense_{2 * (self.n_heads - 1)}"]
            unfrozen_params["params"][f"Dense_{2 * idx_head + 1}"] = params["params"][
                f"Dense_{2 * (self.n_heads - 1) + 1}"
            ]

        return FrozenDict(unfrozen_params)


# class FullyConnectedMultiQNet(hk.Module):
#     def __init__(
#         self,
#         n_heads: int,
#         shared_layers_dimension: list,
#         layers_dimension: list,
#         zero_initializer: bool,
#         n_actions: int,
#     ) -> None:
#         self.n_heads = n_heads
#         super().__init__(name="FullyConnectedMultiQNet")
#         if zero_initializer:
#             initializer = hk.initializers.Constant(0)
#         else:
#             initializer = hk.initializers.TruncatedNormal()
#         self.n_actions = n_actions

#         shared_layers_first_head_ = []
#         shared_layers_other_heads_ = []
#         for idx_layer, shared_layer_dimension in enumerate(shared_layers_dimension):
#             shared_layers_first_head_.extend(
#                 [hk.Linear(shared_layer_dimension, name=f"shared_first_head_linear_{idx_layer}"), jax.nn.relu]
#             )
#             shared_layers_other_heads_.extend(
#                 [hk.Linear(shared_layer_dimension, name=f"shared_other_heads_linear_{idx_layer}"), jax.nn.relu]
#             )
#         self.shared_layers_first_head = hk.Sequential(shared_layers_first_head_, name="shared_layers_first_head")
#         self.shared_layers_other_heads = hk.Sequential(shared_layers_other_heads_, name="shared_layers")

#         self.heads = []
#         for idx_head in range(self.n_heads):
#             head_ = []
#             for idx_layer, layer_dimension in enumerate(layers_dimension):
#                 head_.extend([hk.Linear(layer_dimension, name=f"head_{idx_head}_linear_{idx_layer}"), jax.nn.relu])
#             head_.append(hk.Linear(self.n_actions, w_init=initializer, name=f"head_{idx_head}_linear_last"))
#             self.heads.append(hk.Sequential(head_, name=f"head_{idx_head}"))

#     def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
#         input = state
#         input = jnp.atleast_2d(input)
#         output = jnp.zeros((input.shape[0], self.n_heads, self.n_actions))

#         shared_input_first_head = self.shared_layers_first_head(input)
#         output = output.at[:, 0].set(self.heads[0](shared_input_first_head))

#         shared_input_other_heads = self.shared_layers_other_heads(input)
#         for idx_head in range(1, self.n_heads):
#             output = output.at[:, idx_head].set(self.heads[idx_head](shared_input_other_heads))

#         return output


# class FullyConnectediDQN(iDQN):
#     def __init__(
#         self,
#         importance_iteration: jnp.ndarray,
#         state_shape: list,
#         n_actions: int,
#         gamma: float,
#         network_key: jax.random.PRNGKeyArray,
#         shared_layers_dimension: list,
#         layers_dimension: list,
#         zero_initializer: bool,
#         learning_rate: float,
#     ) -> None:
#         self.n_shared_layers = len(shared_layers_dimension)
#         self.n_layers_head = len(layers_dimension)

#         def network(state: jnp.ndarray) -> jnp.ndarray:
#             return FullyConnectedMultiQNet(
#                 len(importance_iteration) + 1, shared_layers_dimension, layers_dimension, zero_initializer, n_actions
#             )(state)

#         super().__init__(
#             importance_iteration, state_shape, n_actions, network, gamma, network, network_key, learning_rate
#         )

#     # This should go to base_q
#     def move_forward(self, params: hk.Params) -> hk.Params:
#         raise NotImplementedError
#         # # Get random params
#         # random_params = self.random_init_params()
#         # for idx_layer in range(self.n_shared_layers):
#         #     random_params[f"FullyConnectedNet/~/shared_first_head_linear_{idx_layer}"] = {"w": 0, "b": 0}
#         # for idx_layer in range(self.n_layers_head):
#         #     random_params[f"FullyConnectedNet/~/head_0_linear_{idx_layer}"] = {"w": 0, "b": 0}
#         # random_params[f"FullyConnectedNet/~/head_0_linear_last"] = {"w": 0, "b": 0}

#         # # The shared params of the first head takes the shared params of the other heads
#         # for idx_layer in range(self.n_shared_layers):
#         #     params[f"FullyConnectedNet/~/shared_first_head_linear_{idx_layer}"] = add_noise(
#         #         params[f"FullyConnectedNet/~/shared_other_heads_linear_{idx_layer}"],
#         #         random_params[f"FullyConnectedNet/~/shared_first_head_linear_{idx_layer}"],
#         #     )

#         # # Each head takes the params of the last head with some noise
#         # for idx_head in range(self.n_heads):
#         #     for idx_layer in range(self.n_layers_head):
#         #         params[f"FullyConnectedNet/~/head_{idx_head}_linear_{idx_layer}"] = add_noise(
#         #             params[f"FullyConnectedNet/~/head_{self.n_heads - 1}_linear_{idx_layer}"],
#         #             random_params[f"FullyConnectedNet/~/head_{idx_head}_linear_{idx_layer}"],
#         #         )
#         #     params[f"FullyConnectedNet/~/head_{idx_head}_linear_last"] = add_noise(
#         #         params[f"FullyConnectedNet/~/head_{self.n_heads - 1}_linear_last"],
#         #         random_params[f"FullyConnectedNet/~/head_{idx_head}_linear_last"],
#         #     )

#         # return params
