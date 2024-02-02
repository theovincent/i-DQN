from typing import Tuple
from flax.core import FrozenDict
from functools import partial
import flax.linen as nn
import jax
import jax.numpy as jnp

from idqn.networks.base import BaseSingleQ
from idqn.networks.architectures.base import scale
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

    def apply(
        self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKeyArray, n_quantiles: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.network.apply(params, scale(state), key, n_quantiles)

    def compute_target(
        self, params: FrozenDict, sample: Tuple[jnp.ndarray], key: jax.random.PRNGKeyArray
    ) -> jnp.ndarray:
        # We need to n_quantiles_target quantiles for the next value function
        # and n_quantiles_policy quantiles for the next action
        # output (n_quantiles_policy + n_quantiles_target, n_actions)
        next_q_policy_quantiles_quantiles, _ = self.apply(
            params, sample[IDX_RB["next_state"]], key, self.n_quantiles_policy + self.n_quantiles_target
        )

        # output (n_actions)
        next_q_policy_values = jnp.mean(next_q_policy_quantiles_quantiles[: self.n_quantiles_policy], axis=0)
        next_action = jnp.argmax(next_q_policy_values)

        # output (n_quantiles_target)
        return (
            sample[IDX_RB["reward"]]
            + (1 - sample[IDX_RB["terminal"]])
            * self.cumulative_gamma
            * next_q_policy_quantiles_quantiles[self.n_quantiles_policy :, next_action]
        )

    def loss(
        self, params: FrozenDict, params_target: FrozenDict, sample: Tuple[jnp.ndarray], key: jax.random.PRNGKeyArray
    ) -> jnp.float32:
        # output (n_quantiles_target)
        targets = self.compute_target(params_target, sample, jax.random.split(key)[1])

        # output (n_quantiles, n_actions) | (n_quantiles)
        q_quantiles_values, quantiles = self.apply(params, sample[IDX_RB["state"]], key, self.n_quantiles)

        # output (n_quantiles)
        q_quantiles = q_quantiles_values[:, sample[IDX_RB["action"]]]

        # cross difference
        # output (n_quantiles_target, n_quantiles)
        bellman_errors = targets[:, jnp.newaxis] - q_quantiles[jnp.newaxis]

        abs_bellman_errors_mask_low = jax.lax.stop_gradient((jnp.abs(bellman_errors) <= 1).astype(jnp.float32))
        huber_losses_quadratic_case = abs_bellman_errors_mask_low * 0.5 * bellman_errors**2
        abs_bellman_errors_mask_high = jax.lax.stop_gradient((jnp.abs(bellman_errors) > 1).astype(jnp.float32))
        huber_losses_linear_case = abs_bellman_errors_mask_high * (jnp.abs(bellman_errors) - 0.5)
        # output (n_quantiles_target, n_quantiles)
        huber_losses = huber_losses_quadratic_case + huber_losses_linear_case

        bellman_errors_mask_low = jax.lax.stop_gradient(bellman_errors < 0).astype(jnp.float32)

        # mapping over the target quantiles
        # output (n_quantiles_target, n_quantiles)
        quantile_losses = jax.vmap(
            lambda quantile, bellman_error_mask_low, huber_loss: jnp.abs(quantile - bellman_error_mask_low)
            * huber_loss,
            in_axes=(None, 0, 0),
        )(quantiles, bellman_errors_mask_low, huber_losses)

        # sum over the quantiles and mean over the target quantiles
        return jnp.mean(jnp.sum(quantile_losses, axis=1))

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKeyArray) -> jnp.int8:
        # output (n_quantiles_policy, n_actions)
        q_quantiles, _ = self.network.apply(params, scale(state), key, self.n_quantiles_policy)
        q_values = jnp.mean(q_quantiles, axis=0)

        return jnp.argmax(q_values).astype(jnp.int8)
