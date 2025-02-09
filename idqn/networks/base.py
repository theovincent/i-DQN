from typing import Tuple
from flax.core import FrozenDict
from functools import partial
import flax.linen as nn
import optax
import jax
import jax.numpy as jnp

from idqn.sample_collection.replay_buffer import ReplayBuffer
from idqn.utils.pickle import save_pickled_data


class BaseQ:
    def __init__(
        self,
        q_inputs: dict,
        n_actions: int,
        cumulative_gamma: float,
        network: nn.Module,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
        epsilon_optimizer: float,
        n_training_steps_per_online_update: int,
    ) -> None:
        self.n_actions = n_actions
        self.cumulative_gamma = cumulative_gamma
        self.network = network
        self.network_key = network_key
        self.params = self.network.init(self.network_key, **q_inputs)
        self.target_params = self.params
        self.n_training_steps_per_online_update = n_training_steps_per_online_update

        if learning_rate is not None:
            self.optimizer = optax.adam(learning_rate, eps=epsilon_optimizer)
            self.optimizer_state = self.optimizer.init(self.params)

    def loss_on_batch(
        self, params: FrozenDict, params_target: FrozenDict, samples: Tuple[jnp.ndarray], key: jax.random.PRNGKeyArray
    ) -> jnp.float32:
        return jax.vmap(self.loss, in_axes=(None, None, 0, None))(params, params_target, samples, key).mean()

    @staticmethod
    def metric(error: jnp.ndarray, ord: str) -> jnp.float32:
        if ord == "huber":
            return optax.huber_loss(error, 0)
        elif ord == "1":
            return jnp.abs(error)
        elif ord == "2":
            return jnp.square(error)

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self,
        params: FrozenDict,
        params_target: FrozenDict,
        optimizer_state: Tuple,
        batch_samples: Tuple[jnp.ndarray],
        key: jax.random.PRNGKeyArray,
    ) -> Tuple[FrozenDict, FrozenDict, jnp.float32]:
        loss, grad_loss = jax.value_and_grad(self.loss_on_batch)(params, params_target, batch_samples, key)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer, **kwargs) -> jnp.float32:
        if step % self.n_training_steps_per_online_update == 0:
            self.network_key, key = jax.random.split(self.network_key)
            batch_samples = replay_buffer.sample_transition_batch()

            self.params, self.optimizer_state, loss = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples, key
            )

            return loss
        else:
            return jnp.nan

    def update_target_params(self, step: int) -> None:
        raise NotImplementedError

    @partial(jax.jit, static_argnames="self")
    def random_action(self, key: jax.random.PRNGKeyArray) -> jnp.int8:
        return jax.random.choice(key, jnp.arange(self.n_actions)).astype(jnp.int8)

    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKeyArray) -> jnp.int8:
        raise NotImplementedError

    def save(self, path: str) -> None:
        save_pickled_data(path + "_online_params", self.params)


class BaseSingleQ(BaseQ):
    def __init__(
        self,
        q_inputs: dict,
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
            q_inputs,
            n_actions,
            cumulative_gamma,
            network,
            network_key,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
        )

        self.n_training_steps_per_target_update = n_training_steps_per_target_update

    def update_target_params(self, step: int) -> None:
        if step % self.n_training_steps_per_target_update == 0:
            self.target_params = self.params


class BaseIteratedQ(BaseQ):
    def __init__(
        self,
        n_heads: int,
        q_inputs: dict,
        n_actions: int,
        cumulative_gamma: float,
        network: nn.Module,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
        epsilon_optimizer: float,
        n_training_steps_per_online_update: int,
        n_training_steps_per_target_update: int,
        n_training_steps_per_rolling_step: int,
    ) -> None:
        super().__init__(
            q_inputs,
            n_actions,
            cumulative_gamma,
            network,
            network_key,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
        )
        self.n_heads = n_heads
        self.n_training_steps_per_target_update = n_training_steps_per_target_update
        self.n_training_steps_per_rolling_step = n_training_steps_per_rolling_step

    def random_head(self, key: jax.random.PRNGKeyArray, head_probability: jnp.ndarray) -> jnp.int8:
        return jax.random.choice(key, jnp.arange(self.n_heads), p=head_probability)

    def update_target_params(self, step: int) -> None:
        if (step % self.n_training_steps_per_target_update == 0) or (
            step % self.n_training_steps_per_rolling_step == 0
        ):
            self.target_params = self.params

    @partial(jax.jit, static_argnames="self")
    def rolling_step(self, params: FrozenDict) -> FrozenDict:
        return self.network.rolling_step(params)

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer, **kwargs) -> jnp.float32:
        if step % self.n_training_steps_per_rolling_step == 0:
            self.params = self.rolling_step(self.params)

        return super().update_online_params(step, replay_buffer, **kwargs)
