from typing import Tuple
from flax.core import FrozenDict
from functools import partial
import flax.linen as nn
import jax
import jax.numpy as jnp

from idqn.networks.base import BaseSingleQ
from idqn.networks.architectures.base import scale, preprocessor
from idqn.sample_collection import IDX_RB


class IQN(BaseSingleQ):
    def __init__(
        self,
        state_shape: list,
        n_actions: int,
        cumulative_gamma: float,
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
            cumulative_gamma,
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

    def apply_n_quantiles(
        self, params: FrozenDict, states: jnp.ndarray, key: jax.random.PRNGKeyArray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return jax.vmap(self.network.apply, in_axes=(None, 0, None, None))(
            params, preprocessor(states), key, self.n_quantiles
        )

    def apply_n_quantiles_target(
        self, params: FrozenDict, states: jnp.ndarray, key: jax.random.PRNGKeyArray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        We compute n_quantiles_policy + n_quantiles_target so that we use compute the convolution layers once
        (they do not depend on the number of quantiles).
        """
        return jax.vmap(self.network.apply, in_axes=(None, 0, None, None))(
            params, preprocessor(states), key, self.n_quantiles_policy + self.n_quantiles_target
        )

    @staticmethod
    def augment_samples(samples: Tuple[jnp.ndarray], **kwargs) -> Tuple[jnp.ndarray]:
        key, next_key = jax.random.split(kwargs.get("key"))
        samples += (key, next_key)

        return samples

    def compute_target(self, params: FrozenDict, samples: Tuple[jnp.ndarray]) -> jnp.ndarray:
        # We need to n_quantiles_target quantiles for the next value function
        # and n_quantiles_policy quantiles for the next action
        # output (batch_size, n_quantiles_policy + n_quantiles_target, n_actions)
        next_q_policy_quantiles_quantiles, _ = self.apply_n_quantiles_target(
            params, samples[IDX_RB["next_state"]], samples[IDX_RB["next_key"]]
        )

        # output (batch_size, n_actions)
        next_q_policy_values = jnp.mean(next_q_policy_quantiles_quantiles[:, : self.n_quantiles_policy], axis=1)
        next_actions = jnp.argmax(next_q_policy_values, axis=1)  # output (batch_size)

        # mapping over the states
        return jax.vmap(
            lambda reward, terminal, next_q_quantiles, action: reward
            + (1 - terminal) * self.cumulative_gamma * next_q_quantiles[:, action]
        )(
            samples[IDX_RB["reward"]],
            samples[IDX_RB["terminal"]],
            next_q_policy_quantiles_quantiles[:, self.n_quantiles_policy :],
            next_actions,
        )  # output (batch_size, n_quantiles_target)

    def loss(self, params: FrozenDict, params_target: FrozenDict, samples: Tuple[jnp.ndarray]) -> jnp.float32:
        # output (batch_size, n_quantiles_target)
        targets = self.compute_target(params_target, samples)

        # output (batch_size, n_quantiles, n_actions) | (batch_size, n_quantiles)
        q_quantiles, quantiles = self.apply_n_quantiles(params, samples[IDX_RB["state"]], samples[IDX_RB["key"]])

        # mapping over the states
        # output (batch_size, n_quantiles)
        predictions = jax.vmap(lambda quantiles_actions, action: quantiles_actions[:, action])(
            q_quantiles, samples[IDX_RB["action"]]
        )

        # cross difference
        # output (batch_size, n_quantiles_target, n_quantiles)
        bellman_errors = targets[:, :, jnp.newaxis] - predictions[:, jnp.newaxis]
        abs_bellman_errors_mask_low = jax.lax.stop_gradient((jnp.abs(bellman_errors) <= 1).astype(jnp.float32))
        abs_bellman_errors_mask_high = jax.lax.stop_gradient((jnp.abs(bellman_errors) > 1).astype(jnp.float32))
        bellman_errors_mask_low = jax.lax.stop_gradient(bellman_errors < 0).astype(jnp.float32)

        huber_losses_quadratic_case = abs_bellman_errors_mask_low * 0.5 * bellman_errors**2
        huber_losses_linear_case = abs_bellman_errors_mask_high * (jnp.abs(bellman_errors) - 0.5)
        # output (batch_size, n_quantiles_target, n_quantiles)
        huber_losses = huber_losses_quadratic_case + huber_losses_linear_case

        # mapping over the target quantiles
        # output (batch_size, n_quantiles_target, n_quantiles)
        quantile_losses = jax.vmap(
            lambda quantile, bellman_error_mask_low, huber_loss: jnp.abs(quantile - bellman_error_mask_low)
            * huber_loss,
            (None, 1, 1),
            1,
        )(quantiles, bellman_errors_mask_low, huber_losses)

        # sum over the quantiles and mean over the target quantiles and the states
        return jnp.mean(jnp.sum(quantile_losses, axis=2))

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, **kwargs) -> jnp.int8:
        q_quantiles, _ = self.network.apply(params, scale(state), kwargs.get("key"), self.n_quantiles_policy)
        q_values = jnp.mean(q_quantiles, axis=0)

        return jnp.argmax(q_values).astype(jnp.int8)
