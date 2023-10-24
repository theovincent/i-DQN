from typing import Tuple
from flax.core import FrozenDict
from functools import partial
import flax.linen as nn
import jax
import jax.numpy as jnp

from idqn.networks.base_q import BaseIteratedQ
from idqn.sample_collection import IDX_RB


class iIQN(BaseIteratedQ):
    def __init__(
        self,
        n_heads: int,
        state_shape: list,
        n_actions: int,
        gamma: float,
        network: nn.Module,
        network_key: jax.random.PRNGKeyArray,
        head_behaviorial_probability: jnp.ndarray,
        learning_rate: float,
        epsilon_optimizer: float,
        n_training_steps_per_online_update: int,
        n_training_steps_per_target_update: int,
        n_training_steps_per_rolling_step: int,
        n_quantiles_policy: int,
        n_quantiles: int,
        n_quantiles_target: int,
    ) -> None:
        super().__init__(
            n_heads,
            {"state": jnp.zeros(state_shape, dtype=jnp.float32), "key": jax.random.PRNGKey(0), "n_quantiles": 32},
            n_actions,
            gamma,
            network,
            network_key,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
            n_training_steps_per_rolling_step,
        )
        self.head_behaviorial_probability = head_behaviorial_probability
        self.n_quantiles_policy = n_quantiles_policy
        self.n_quantiles = n_quantiles
        self.n_quantiles_target = n_quantiles_target

    @partial(jax.jit, static_argnames="self")
    def apply_n_quantiles_policy(
        self, params: FrozenDict, states: jnp.ndarray, key: jax.random.PRNGKeyArray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.network.apply(params, states, key, self.n_quantiles_policy)

    @partial(jax.jit, static_argnames="self")
    def apply_n_quantiles(
        self, params: FrozenDict, states: jnp.ndarray, key: jax.random.PRNGKeyArray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.network.apply_other_heads(params, states, key, self.n_quantiles)

    @partial(jax.jit, static_argnames="self")
    def apply_n_quantiles_target(
        self, params: FrozenDict, states: jnp.ndarray, key: jax.random.PRNGKeyArray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.network.apply(params, states, key, self.n_quantiles_policy + self.n_quantiles_target)

    def best_action_from_head(
        self, params: FrozenDict, idx_head: int, state: jnp.ndarray, key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """This function is supposed to take a single state and not a batch"""
        return jnp.argmax(
            jnp.mean(
                self.network.apply_specific_head(params, idx_head, state, key, self.n_quantiles_policy)[0],
                axis=0,
            )
        ).astype(jnp.int8)

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
        )  # output (batch_size, n_heads, n_quantiles_policy + n_quantiles_target, n_actions)
        next_states_policy_values_actions = jnp.mean(
            next_states_policy_quantiles_quantiles_actions[:, :, : self.n_quantiles_policy], axis=2
        )  # output (batch_size, n_heads, n_actions)
        next_states_action = jnp.argmax(next_states_policy_values_actions, axis=2)  # output (batch_size, n_heads)

        # mapping first over the states and then over the heads
        next_states_quantiles = jax.vmap(jax.vmap(lambda quantiles_actions, action: quantiles_actions[:, action]))(
            next_states_policy_quantiles_quantiles_actions[:, :, self.n_quantiles_policy :], next_states_action
        )  # output (batch_size, n_heads, n_quantiles_target)

        # mapping over the states
        return jax.vmap(
            lambda reward, terminal, next_states_quantiles_: reward
            + (1 - terminal) * self.gamma * next_states_quantiles_,
        )(
            samples[IDX_RB["reward"]],
            samples[IDX_RB["terminal"]],
            next_states_quantiles,
        )  # output (batch_size, n_heads, n_quantiles_target)

    @partial(jax.jit, static_argnames="self")
    def loss(self, params: FrozenDict, params_target: FrozenDict, samples: Tuple[jnp.ndarray]) -> jnp.float32:
        targets = self.compute_target(params_target, samples)  # output (batch_size, n_heads, n_quantiles_target)

        states_quantiles_actions, quantiles = self.apply_n_quantiles(
            params, samples[IDX_RB["state"]], samples[IDX_RB["key"]]
        )  # output (batch_size, n_heads - 1, n_quantiles, n_actions) | (batch_size, n_quantiles)
        # mapping over the states
        predictions = jax.vmap(lambda quantiles_actions, action: quantiles_actions[:, :, action])(
            states_quantiles_actions, samples[IDX_RB["action"]]
        )  # output (batch_size, n_heads - 1, n_quantiles)

        # cross difference
        bellman_errors = (
            targets[:, :-1, :, jnp.newaxis] - predictions[:, :, jnp.newaxis]
        )  # output (batch_size, n_heads - 1, n_quantiles_target, n_quantiles)
        abs_bellman_errors_mask_low = jax.lax.stop_gradient((jnp.abs(bellman_errors) <= 1).astype(jnp.float32))
        abs_bellman_errors_mask_high = jax.lax.stop_gradient((jnp.abs(bellman_errors) > 1).astype(jnp.float32))
        bellman_errors_mask_low = jax.lax.stop_gradient(bellman_errors < 0).astype(jnp.float32)

        huber_losses_quadratic_case = abs_bellman_errors_mask_low * 0.5 * bellman_errors**2
        huber_losses_linear_case = abs_bellman_errors_mask_high * (jnp.abs(bellman_errors) - 0.5)
        huber_losses = (
            huber_losses_quadratic_case + huber_losses_linear_case
        )  # output (batch_size, n_heads - 1, n_quantiles_target, n_quantiles)

        # mapping first over the heads and then over the target quantiles
        quantile_losses = jax.vmap(
            jax.vmap(
                lambda quantile, bellman_error_mask_low, huber_losses_: jnp.abs(quantile - bellman_error_mask_low)
                * huber_losses_,
                (None, 1, 1),
                1,
            ),
            (None, 1, 1),
            1,
        )(
            quantiles, bellman_errors_mask_low, huber_losses
        )  # output (batch_size, n_heads - 1, n_quantiles_target, n_quantiles)

        # sum over the quantiles and mean over the target quantiles, the heads and the states
        return jnp.mean(jnp.sum(quantile_losses, axis=3))

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, **kwargs) -> jnp.int8:
        idx_head = self.random_head(kwargs.get("key"), self.head_behaviorial_probability)

        return self.best_action_from_head(params, idx_head, state, kwargs.get("key"))
