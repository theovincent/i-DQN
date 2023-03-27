from typing import Tuple
from flax.core import FrozenDict
from functools import partial
import flax.linen as nn
import optax
import jax
import jax.numpy as jnp

from idqn.sample_collection.replay_buffer import ReplayBuffer
from idqn.utils.pickle import load_pickled_data, save_pickled_data


class BaseQ:
    def __init__(
        self,
        state_shape: list,
        n_actions: int,
        gamma: float,
        network: nn.Module,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
        n_training_steps_per_online_update: int,
    ) -> None:
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.network = network
        self.network_key = network_key
        self.params = self.network.init(self.network_key, state=jnp.zeros(self.state_shape, dtype=jnp.float32))
        self.target_params = self.params
        self.n_training_steps_per_online_update = n_training_steps_per_online_update

        self.loss_and_grad = jax.jit(jax.value_and_grad(self.loss))

        if learning_rate is not None:
            self.learning_rate = learning_rate
            self.optimizer = optax.adam(self.learning_rate, eps=1.5e-4)
            self.optimizer_state = self.optimizer.init(self.params)

    @partial(jax.jit, static_argnames="self")
    def __call__(self, params: FrozenDict, states: jnp.ndarray) -> jnp.ndarray:
        # "jnp.atleast{batch_dims}d"
        inputs = jnp.array(states, ndmin=len(self.state_shape) + 1)

        return jax.vmap(self.network.apply, in_axes=[None, 0])(params, inputs)

    def loss(self, params: FrozenDict, params_target: FrozenDict, samples: FrozenDict, ord: int = 2) -> jnp.float32:
        raise NotImplementedError

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self, params: FrozenDict, params_target: FrozenDict, optimizer_state: Tuple, batch_samples: jnp.ndarray
    ) -> Tuple[FrozenDict, FrozenDict, jnp.float32]:
        loss, grad_loss = self.loss_and_grad(params, params_target, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer, key: jax.random.PRNGKeyArray) -> jnp.float32:
        if step % self.n_training_steps_per_online_update == 0:
            batch_samples = replay_buffer.sample_random_batch(key)

            self.params, self.optimizer_state, loss = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples
            )

            return loss
        else:
            return jnp.nan

    def update_target_params(self, step: int) -> None:
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
    def random_action(self, key: jax.random.PRNGKeyArray) -> jnp.int8:
        return jax.random.choice(key, jnp.arange(self.n_actions)).astype(jnp.int8)

    def best_action(self, key: jax.random.PRNGKey, q_params: FrozenDict, state: jnp.ndarray) -> jnp.int8:
        raise NotImplementedError

    def save(self, path: str, online_params_only: bool = False) -> None:
        save_pickled_data(path + "_online_params", self.params)

        if not online_params_only:
            save_pickled_data(path + "_target_params", self.target_params)
            save_pickled_data(path + "_optimizer", self.optimizer_state)

    def load(self, path: str) -> None:
        self.params = load_pickled_data(path + "_online_params", device_put=True)
        self.target_params = load_pickled_data(path + "_target_params", device_put=True)
        self.optimizer_state = load_pickled_data(path + "_optimizer", device_put=True)


class BaseSingleQ(BaseQ):
    def __init__(
        self,
        state_shape: list,
        n_actions: int,
        gamma: float,
        network: nn.Module,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
        n_training_steps_per_online_update: int,
    ) -> None:
        super().__init__(
            state_shape, n_actions, gamma, network, network_key, learning_rate, n_training_steps_per_online_update
        )

    @partial(jax.jit, static_argnames="self")
    def compute_target(self, params: FrozenDict, samples: FrozenDict) -> jnp.ndarray:
        return samples["reward"] + (1 - samples["absorbing"]) * self.gamma * self(params, samples["next_state"]).max(
            axis=1
        )


class DQN(BaseSingleQ):
    def __init__(
        self,
        state_shape: list,
        n_actions: int,
        gamma: float,
        network: nn.Module,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
        n_training_steps_per_online_update: int,
        n_training_steps_per_target_update: int,
    ) -> None:
        super().__init__(
            state_shape, n_actions, gamma, network, network_key, learning_rate, n_training_steps_per_online_update
        )
        self.n_training_steps_per_target_update = n_training_steps_per_target_update

    @partial(jax.jit, static_argnames="self")
    def loss(self, params: FrozenDict, params_target: FrozenDict, samples: FrozenDict) -> jnp.float32:
        targets = self.compute_target(params_target, samples)
        q_states_actions = self(params, samples["state"])
        predictions = jax.vmap(lambda q_state_actions, action: q_state_actions[action])(
            q_states_actions, samples["action"]
        )

        error = predictions - targets
        return self.metric(error, ord="2")

    @partial(jax.jit, static_argnames="self")
    def bellman_error(self, params: FrozenDict, params_target: FrozenDict, samples: FrozenDict) -> jnp.float32:
        targets = self.compute_target(params_target, samples)
        predictions = self(params, samples["state"])[jnp.arange(samples["state"].shape[0]), samples["action"]]

        error = predictions - targets
        return self.metric(error, ord="sum")

    @partial(jax.jit, static_argnames="self")
    def best_action(self, key: jax.random.PRNGKey, q_params: FrozenDict, state: jnp.ndarray) -> jnp.int8:
        # key is not used here
        return jnp.argmax(self(q_params, jnp.array(state, dtype=jnp.float32))[0]).astype(jnp.int8)

    def update_target_params(self, step: int) -> None:
        if step % self.n_training_steps_per_target_update == 0:
            self.target_params = self.params


class BaseMultiHeadQ(BaseQ):
    def __init__(
        self,
        n_heads: int,
        state_shape: list,
        n_actions: int,
        gamma: float,
        network: nn.Module,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
        n_training_steps_per_online_update: int,
    ) -> None:
        self.n_heads = n_heads
        super().__init__(
            state_shape, n_actions, gamma, network, network_key, learning_rate, n_training_steps_per_online_update
        )

    @partial(jax.jit, static_argnames="self")
    def compute_target(self, params: FrozenDict, samples: FrozenDict) -> jnp.ndarray:
        return jnp.repeat(samples["reward"][:, None], self.n_heads, axis=1) + jnp.repeat(
            1 - samples["absorbing"][:, None], self.n_heads, axis=1
        ) * self.gamma * self(params, samples["next_state"]).max(axis=2)

    @partial(jax.jit, static_argnames="self")
    def random_head(self, key: jax.random.PRNGKeyArray, head_probability: jnp.ndarray) -> jnp.int8:
        return jax.random.choice(key, jnp.arange(self.n_heads), p=head_probability)


class iDQN(BaseMultiHeadQ):
    def __init__(
        self,
        importance_iteration: jnp.ndarray,
        state_shape: list,
        n_actions: int,
        gamma: float,
        network: nn.Module,
        network_key: jax.random.PRNGKeyArray,
        head_behaviorial_probability: jnp.ndarray,
        learning_rate: float,
        n_training_steps_per_online_update: int,
        n_training_steps_per_target_update: int,
        n_training_steps_per_head_update: int,
    ) -> None:
        self.importance_iteration = importance_iteration
        self.head_behaviorial_probability = head_behaviorial_probability
        self.n_training_steps_per_target_update = n_training_steps_per_target_update
        self.n_training_steps_per_head_update = n_training_steps_per_head_update

        super().__init__(
            len(importance_iteration) + 1,
            state_shape,
            n_actions,
            gamma,
            network,
            network_key,
            learning_rate,
            n_training_steps_per_online_update,
        )

    def update_heads(self, params: FrozenDict) -> FrozenDict:
        raise NotImplementedError

    @partial(jax.jit, static_argnames="self")
    def loss(self, params: FrozenDict, params_target: FrozenDict, samples: FrozenDict) -> jnp.float32:
        targets = self.compute_target(params_target, samples)[:, :-1]
        predictions = self(params, samples["state"])[jnp.arange(samples["state"].shape[0]), 1:, samples["action"]]

        error = (predictions - targets) * jnp.repeat(self.importance_iteration[None, :], targets.shape[0], axis=0)
        return self.metric(error, ord="2")

    @partial(jax.jit, static_argnames="self")
    def bellman_error(self, params: FrozenDict, params_target: FrozenDict, samples: FrozenDict) -> jnp.float32:
        targets = self.compute_target(params_target, samples)
        predictions = self(params, samples["state"])[jnp.arange(samples["state"].shape[0]), :, samples["action"]]

        error = predictions - targets
        return self.metric(error, ord="sum")

    @partial(jax.jit, static_argnames="self")
    def best_action(self, key: jax.random.PRNGKey, q_params: FrozenDict, state: jnp.ndarray) -> jnp.int8:
        idx_head = self.random_head(key, self.head_behaviorial_probability)

        return jnp.argmax(self(q_params, jnp.array(state, dtype=jnp.float32))[0, idx_head]).astype(jnp.int8)

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer, key: jax.random.PRNGKeyArray) -> jnp.float32:
        loss = super().update_online_params(step, replay_buffer, key)

        if step % self.n_training_steps_per_head_update == 0:
            self.params = self.update_heads(self.params)

        return loss

    def update_target_params(self, step: int) -> None:
        if (step % self.n_training_steps_per_target_update == 0) or (step % self.n_training_steps_per_head_update == 0):
            self.target_params = self.params
