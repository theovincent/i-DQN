from typing import Tuple
from flax.core import FrozenDict
from functools import partial
import flax.linen as nn
import jax
import jax.numpy as jnp

from idqn.networks.base_q import BaseSingleQ
from idqn.sample_collection import IDX_RB


class IQN(BaseSingleQ):
    def __init__(
        self,
        state_shape: list,
        n_actions: int,
        gamma: float,
        network: nn.Module,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
        epsilon_optimizer: float,
        n_training_steps_per_online_update: int,
        n_training_steps_per_target_update: int,
    ) -> None:
        super().__init__(
            {"state": jnp.zeros(state_shape, dtype=jnp.float32), "key": jax.random.PRNGKey(0), "n_quantiles": 32},
            n_actions,
            gamma,
            network,
            network_key,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
        )
        self.n_quantiles_policy = 32
        self.n_quantiles = 64
        self.n_quantiles_target = 64

    @partial(jax.jit, static_argnames="self")
    def apply_n_quantiles_policy(
        self, params: FrozenDict, states: jnp.ndarray, key: jax.random.PRNGKeyArray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.network.apply(params, states, key, self.n_quantiles_policy)

    @partial(jax.jit, static_argnames="self")
    def apply_n_quantiles(
        self, params: FrozenDict, states: jnp.ndarray, key: jax.random.PRNGKeyArray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.network.apply(params, states, key, self.n_quantiles)

    @partial(jax.jit, static_argnames="self")
    def apply_n_quantiles_target(
        self, params: FrozenDict, states: jnp.ndarray, key: jax.random.PRNGKeyArray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        We compute n_quantiles_policy + n_quantiles_target so that we use compute the convolution layers once
        (they do not depend on the number of quantiles).
        """
        return self.network.apply(params, states, key, self.n_quantiles_policy + self.n_quantiles_target)

    @staticmethod
    @partial(jax.jit, static_argnames="self")
    def augment_samples(samples: Tuple[jnp.ndarray], **kwargs) -> Tuple[jnp.ndarray]:
        key, next_key = jax.random.split(kwargs.get("key"))
        samples += (key, next_key)

        return samples

    @partial(jax.jit, static_argnames="self")
    def compute_target(self, params: FrozenDict, samples: Tuple[jnp.ndarray]) -> jnp.ndarray:
        next_states_policy_quantiles_quantiles_actions, _ = self.apply_n_quantiles_target(
            params, samples[IDX_RB["next_state"]], samples[IDX_RB["next_key"]]
        )  # output (batch_size, n_quantiles_policy + n_quantiles_target, n_actions)
        next_states_policy_values_actions = jnp.mean(
            next_states_policy_quantiles_quantiles_actions[:, : self.n_quantiles_policy], axis=1
        )  # output (batch_size, n_actions)
        next_states_action = jnp.argmax(next_states_policy_values_actions, axis=1)  # output (batch_size)

        # mapping over the states
        return jax.vmap(
            lambda reward, terminal, next_states_quantiles_actions_, action: reward
            + (1 - terminal) * self.gamma * next_states_quantiles_actions_[:, action]
        )(
            samples[IDX_RB["reward"]],
            samples[IDX_RB["terminal"]],
            next_states_policy_quantiles_quantiles_actions[:, self.n_quantiles_policy :],
            next_states_action,
        )  # output (batch_size, n_quantiles_target)

    @partial(jax.jit, static_argnames="self")
    def loss(self, params: FrozenDict, params_target: FrozenDict, samples: Tuple[jnp.ndarray]) -> jnp.float32:
        targets = self.compute_target(params_target, samples)  # output (batch_size, n_quantiles_target)

        states_quantiles_actions, quantiles = self.apply_n_quantiles(
            params, samples[IDX_RB["state"]], samples[IDX_RB["key"]]
        )  # output (batch_size, n_quantiles, n_actions) | (batch_size, n_quantiles)
        # mapping over the states
        predictions = jax.vmap(lambda quantiles_actions, action: quantiles_actions[:, action])(
            states_quantiles_actions, samples[IDX_RB["action"]]
        )  # output (batch_size, n_quantiles)

        # cross difference
        bellman_errors = (
            targets[:, :, jnp.newaxis] - predictions[:, jnp.newaxis]
        )  # output (batch_size, n_quantiles_target, n_quantiles)
        abs_bellman_errors_mask_low = jax.lax.stop_gradient((jnp.abs(bellman_errors) <= 1).astype(jnp.float32))
        abs_bellman_errors_mask_high = jax.lax.stop_gradient((jnp.abs(bellman_errors) > 1).astype(jnp.float32))
        bellman_errors_mask_low = jax.lax.stop_gradient(bellman_errors < 0).astype(jnp.float32)

        huber_losses_quadratic_case = abs_bellman_errors_mask_low * 0.5 * bellman_errors**2
        huber_losses_linear_case = abs_bellman_errors_mask_high * (jnp.abs(bellman_errors) - 0.5)
        huber_losses = (
            huber_losses_quadratic_case + huber_losses_linear_case
        )  # output (batch_size, n_quantiles_target, n_quantiles)

        # mapping over the target quantiles
        quantile_losses = jax.vmap(
            lambda quantile, bellman_error_mask_low, huber_loss: jnp.abs(quantile - bellman_error_mask_low)
            * huber_loss,
            (None, 1, 1),
            1,
        )(
            quantiles, bellman_errors_mask_low, huber_losses
        )  # output (batch_size, n_quantiles_target, n_quantiles)

        # sum over the quantiles and mean over the target quantiles and the states
        return jnp.mean(jnp.sum(quantile_losses, axis=2))

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, **kwargs) -> jnp.int8:
        state_quantiles, _ = self.apply_n_quantiles_policy(
            params, jnp.array(state, dtype=jnp.float32), kwargs.get("key")
        )
        state_values = jnp.mean(state_quantiles, axis=(0, 1))

        return jnp.argmax(state_values).astype(jnp.int8)
