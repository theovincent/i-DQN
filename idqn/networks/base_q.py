from typing import Tuple
from functools import partial
import haiku as hk
import optax
import jax
import jax.numpy as jnp


class BaseQ:
    def __init__(
        self,
        state_shape: list,
        n_actions: int,
        gamma: float,
        network: hk.Module,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
    ) -> None:
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.network = hk.without_apply_rng(hk.transform(network))
        self.network_key = network_key
        self.params = self.network.init(rng=self.network_key, state=jnp.zeros(self.state_shape, dtype=jnp.float32))
        self.target_params = self.params

        self.loss_and_grad = jax.jit(jax.value_and_grad(self.loss))

        if learning_rate is not None:
            self.learning_rate = learning_rate
            self.optimizer = optax.adam(self.learning_rate, eps=0.0003125)
            self.optimizer_state = self.optimizer.init(self.params)

    @partial(jax.jit, static_argnames="self")
    def __call__(self, params: hk.Params, states: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply(params, states)

    def loss(self, params: hk.Params, params_target: hk.Params, samples: dict, ord: int = 2) -> jnp.float32:
        raise NotImplementedError

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self, params: hk.Params, params_target: hk.Params, optimizer_state: tuple, batch_samples: jnp.ndarray
    ) -> Tuple[hk.Params, dict, jnp.float32]:
        loss, grad_loss = self.loss_and_grad(params, params_target, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

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

    def best_action(self, key: jax.random.PRNGKey, q_params: hk.Params, state: jnp.ndarray) -> jnp.int8:
        raise NotImplementedError

    def update_target_params(self, step: int) -> None:
        raise NotImplementedError


class BaseSingleQ(BaseQ):
    def __init__(
        self,
        state_shape: list,
        n_actions: int,
        gamma: float,
        network: hk.Module,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
    ) -> None:
        super().__init__(state_shape, n_actions, gamma, network, network_key, learning_rate)

    @partial(jax.jit, static_argnames="self")
    def compute_target(self, params: hk.Params, samples: dict) -> jnp.ndarray:
        return samples["reward"] + (1 - samples["absorbing"]) * self.gamma * self(params, samples["next_state"]).max(
            axis=1
        )


class DQN(BaseSingleQ):
    def __init__(
        self,
        state_shape: list,
        n_actions: int,
        gamma: float,
        network: hk.Module,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
        n_gradient_steps_per_target_update: int,
    ) -> None:
        super().__init__(state_shape, n_actions, gamma, network, network_key, learning_rate)
        self.n_gradient_steps_per_target_update = n_gradient_steps_per_target_update

    @partial(jax.jit, static_argnames="self")
    def loss(self, params: hk.Params, params_target: hk.Params, samples: dict) -> jnp.float32:
        targets = self.compute_target(params_target, samples)
        predictions = self(params, samples["state"])[jnp.arange(samples["state"].shape[0]), samples["action"]]

        error = predictions - targets
        return self.metric(error, ord="2")

    @partial(jax.jit, static_argnames="self")
    def bellman_error(self, params: hk.Params, params_target: hk.Params, samples: dict) -> jnp.float32:
        targets = self.compute_target(params_target, samples)
        predictions = self(params, samples["state"])[jnp.arange(samples["state"].shape[0]), samples["action"]]

        error = predictions - targets
        return self.metric(error, ord="sum")

    @partial(jax.jit, static_argnames="self")
    def best_action(self, key: jax.random.PRNGKey, q_params: hk.Params, state: jnp.ndarray) -> jnp.int8:
        # key is not used here
        return jnp.argmax(self(q_params, jnp.array(state, dtype=jnp.float32))[0]).astype(jnp.int8)

    def update_target_params(self, step: int) -> None:
        if step % self.n_gradient_steps_per_target_update == 0:
            self.target_params = self.params


class BaseMultiHeadQ(BaseQ):
    def __init__(
        self,
        n_heads: int,
        n_actions: int,
        state_shape: list,
        gamma: float,
        network: hk.Module,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
    ) -> None:
        self.n_heads = n_heads
        super().__init__(state_shape, n_actions, gamma, network, network_key, learning_rate)

    @partial(jax.jit, static_argnames="self")
    def compute_target(self, params: hk.Params, samples: dict) -> jnp.ndarray:
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
        network: hk.Module,
        network_key: jax.random.PRNGKeyArray,
        head_behaviorial_probability: jnp.ndarray,
        learning_rate: float,
    ) -> None:
        self.importance_iteration = importance_iteration
        self.head_behaviorial_probability = head_behaviorial_probability

        super().__init__(
            len(importance_iteration) + 1, state_shape, n_actions, gamma, network, network_key, learning_rate
        )

    def move_forward(self, params: hk.Params) -> hk.Params:
        raise NotImplementedError

    @partial(jax.jit, static_argnames="self")
    def loss(self, params: hk.Params, params_target: hk.Params, samples: dict) -> jnp.float32:
        targets = self.compute_target(params_target, samples)[:, :-1]
        predictions = self(params, samples["state"])[jnp.arange(samples["state"].shape[0]), 1:, samples["action"]]

        error = (predictions - targets) * jnp.repeat(self.importance_iteration[None, :], targets.shape[0], axis=0)
        return self.metric(error, ord="2")

    @partial(jax.jit, static_argnames="self")
    def bellman_error(self, params: hk.Params, params_target: hk.Params, samples: dict) -> jnp.float32:
        targets = self.compute_target(params_target, samples)
        predictions = self(params, samples["state"])[jnp.arange(samples["state"].shape[0]), :, samples["action"]]

        error = predictions - targets
        return self.metric(error, ord="sum")

    @partial(jax.jit, static_argnames="self")
    def best_action(self, key: jax.random.PRNGKey, q_params: hk.Params, state: jnp.ndarray) -> jnp.int8:
        idx_head = self.random_head(key, self.head_behaviorial_probability)

        return jnp.argmax(self(q_params, jnp.array(state, dtype=jnp.float32))[0, idx_head]).astype(jnp.int8)
