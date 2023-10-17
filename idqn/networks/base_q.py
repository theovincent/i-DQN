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
        gamma: float,
        network: nn.Module,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
        epsilon_optimizer: float,
        n_training_steps_per_online_update: int,
    ) -> None:
        self.n_actions = n_actions
        self.gamma = gamma
        self.network = network
        self.network_key = network_key
        self.params = self.network.init(self.network_key, **q_inputs)
        self.target_params = self.params
        self.n_training_steps_per_online_update = n_training_steps_per_online_update

        self.loss_and_grad = jax.jit(jax.value_and_grad(self.loss))

        if learning_rate is not None:
            self.optimizer = optax.adam(learning_rate, eps=epsilon_optimizer)
            self.optimizer_state = self.optimizer.init(self.params)

    def compute_target(self, params: FrozenDict, samples: Tuple[jnp.ndarray]) -> jnp.ndarray:
        raise NotImplementedError

    def loss(
        self, params: FrozenDict, params_target: FrozenDict, samples: Tuple[jnp.ndarray], ord: int = 2
    ) -> jnp.float32:
        raise NotImplementedError

    @staticmethod
    def metric(error: jnp.ndarray, ord: str) -> jnp.float32:
        if ord == "huber":
            return optax.huber_loss(error, 0).mean()
        elif ord == "1":
            return jnp.abs(error).mean()
        elif ord == "2":
            return jnp.square(error).mean()
        elif ord == "sum":
            return jnp.square(error).sum()

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self, params: FrozenDict, params_target: FrozenDict, optimizer_state: Tuple, batch_samples: Tuple[jnp.ndarray]
    ) -> Tuple[FrozenDict, FrozenDict, jnp.float32]:
        loss, grad_loss = self.loss_and_grad(params, params_target, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    def add_keys(self, samples: Tuple[jnp.ndarray]) -> Tuple[jnp.ndarray]:
        return samples

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer) -> jnp.float32:
        if step % self.n_training_steps_per_online_update == 0:
            batch_samples = replay_buffer.sample_transition_batch()
            batch_samples = self.add_keys(batch_samples)

            self.params, self.optimizer_state, loss = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples
            )

            return loss
        else:
            return jnp.nan

    def update_target_params(self, step: int) -> None:
        raise NotImplementedError

    @partial(jax.jit, static_argnames="self")
    def random_action(self, key: jax.random.PRNGKeyArray) -> jnp.int8:
        return jax.random.choice(key, jnp.arange(self.n_actions)).astype(jnp.int8)

    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.int8:
        raise NotImplementedError

    def save(self, path: str) -> None:
        save_pickled_data(path + "_online_params", self.params)


class BaseSingleQ(BaseQ):
    def __init__(
        self,
        q_inputs: dict,
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
            q_inputs,
            n_actions,
            gamma,
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
        gamma: float,
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
            gamma,
            network,
            network_key,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
        )
        self.n_heads = n_heads
        self.n_training_steps_per_target_update = n_training_steps_per_target_update
        self.n_training_steps_per_rolling_step = n_training_steps_per_rolling_step

    @partial(jax.jit, static_argnames="self")
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

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer) -> jnp.float32:
        loss = super().update_online_params(step, replay_buffer)

        if step % self.n_training_steps_per_rolling_step == 0:
            self.params = self.rolling_step(self.params)

        return loss
