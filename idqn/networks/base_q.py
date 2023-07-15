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

    def compute_target(self, params: FrozenDict, samples: FrozenDict) -> jnp.ndarray:
        raise NotImplementedError

    def loss(self, params: FrozenDict, params_target: FrozenDict, samples: FrozenDict, ord: int = 2) -> jnp.float32:
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
        self, params: FrozenDict, params_target: FrozenDict, optimizer_state: Tuple, batch_samples: jnp.ndarray
    ) -> Tuple[FrozenDict, FrozenDict, jnp.float32]:
        loss, grad_loss = self.loss_and_grad(params, params_target, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    def add_keys(self, samples):
        pass

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer, key: jax.random.PRNGKeyArray) -> jnp.float32:
        if step % self.n_training_steps_per_online_update == 0:
            batch_samples = replay_buffer.sample_random_batch(key)
            self.add_keys(batch_samples)

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


class DQN(BaseSingleQ):
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
            {"state": jnp.zeros(state_shape, dtype=jnp.float32)},
            n_actions,
            gamma,
            network,
            network_key,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
        )

    @partial(jax.jit, static_argnames="self")
    def apply(self, params: FrozenDict, states: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply(params, states)

    @partial(jax.jit, static_argnames="self")
    def compute_target(self, params: FrozenDict, samples: FrozenDict) -> jnp.ndarray:
        return samples["reward"] + (1 - samples["absorbing"]) * self.gamma * self.apply(
            params, samples["next_state"]
        ).max(axis=1)

    @partial(jax.jit, static_argnames="self")
    def loss(self, params: FrozenDict, params_target: FrozenDict, samples: FrozenDict) -> jnp.float32:
        targets = self.compute_target(params_target, samples)
        q_states_actions = self.apply(params, samples["state"])

        # mapping over the states
        predictions = jax.vmap(lambda q_state_actions, action: q_state_actions[action])(
            q_states_actions, samples["action"]
        )

        return self.metric(predictions - targets, ord="2")

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.int8:
        # key is not used here
        return jnp.argmax(self.apply(params, jnp.array(state, dtype=jnp.float32))[0]).astype(jnp.int8)


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
        return self.network.apply(params, states, key, self.n_quantiles_target)

    def add_keys(self, samples):
        self.network_key, samples["key"], samples["next_key"], samples["policy_key"] = jax.random.split(
            self.network_key, 4
        )

    @partial(jax.jit, static_argnames="self")
    def compute_target(self, params: FrozenDict, samples: FrozenDict) -> jnp.ndarray:
        next_states_policy_quantiles_actions, _ = self.apply_n_quantiles_policy(
            params, samples["next_state"], samples["policy_key"]
        )  # output (batch_size, n_quantiles_policy, n_actions)
        next_states_policy_values_actions = jnp.mean(
            next_states_policy_quantiles_actions, axis=1
        )  # output (batch_size, n_actions)
        next_states_action = jnp.argmax(next_states_policy_values_actions, axis=1)  # output (batch_size)

        next_states_quantiles_actions, _ = self.apply_n_quantiles_target(
            params, samples["next_state"], samples["next_key"]
        )  # output (batch_size, n_quantiles_target, n_actions)

        # mapping over the states
        next_states_quantiles = jax.vmap(lambda quantiles_actions, action: quantiles_actions[:, action])(
            next_states_quantiles_actions, next_states_action
        )  # output (batch_size, n_quantiles_target)

        # mapping over the states
        return jax.vmap(
            lambda reward, absorbing, next_states_quantiles_: reward
            + (1 - absorbing) * self.gamma * next_states_quantiles_
        )(
            samples["reward"], samples["absorbing"], next_states_quantiles
        )  # output (batch_size, n_quantiles_target)

    @partial(jax.jit, static_argnames="self")
    def loss(self, params: FrozenDict, params_target: FrozenDict, samples: FrozenDict) -> jnp.float32:
        targets = self.compute_target(params_target, samples)  # output (batch_size, n_quantiles_target)

        states_quantiles_actions, quantiles = self.apply_n_quantiles(
            params, samples["state"], samples["key"]
        )  # output (batch_size, n_quantiles, n_actions) | (batch_size, n_quantiles)
        # mapping over the states
        predictions = jax.vmap(lambda quantiles_actions, action: quantiles_actions[:, action])(
            states_quantiles_actions, samples["action"]
        )  # output (batch_size, n_quantiles)

        # cross difference
        bellman_errors = (
            targets[:, :, jnp.newaxis] - predictions[:, jnp.newaxis]
        )  # output (batch_size, n_quantiles_target, n_quantiles)

        huber_losses_quadratic_case = (jnp.abs(bellman_errors) <= 1).astype(jnp.float32) * 0.5 * bellman_errors**2
        huber_losses_linear_case = (jnp.abs(bellman_errors) > 1).astype(jnp.float32) * (jnp.abs(bellman_errors) - 0.5)
        huber_losses = (
            huber_losses_quadratic_case + huber_losses_linear_case
        )  # output (batch_size, n_quantiles_target, n_quantiles)

        # mapping over the target quantiles
        quantile_losses = jax.vmap(
            lambda quantile, bellman_error, huber_loss: jnp.abs(
                quantile - jax.lax.stop_gradient(bellman_error < 0).astype(jnp.float32)
            )
            * huber_loss,
            (None, 1, 1),
            1,
        )(
            quantiles, bellman_errors, huber_losses
        )  # output (batch_size, n_quantiles_target, n_quantiles)

        # sum over the quantiles and mean over the target quantiles and the states
        return jnp.mean(jnp.sum(quantile_losses, axis=2))

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.int8:
        state_quantiles, _ = self.apply_n_quantiles_policy(params, jnp.array(state, dtype=jnp.float32), key)
        state_values = jnp.mean(state_quantiles, axis=(0, 1))

        return jnp.argmax(state_values).astype(jnp.int8)


class BaseMultiHeadQ(BaseQ):
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
        n_training_steps_per_head_update: int,
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
        self.n_training_steps_per_head_update = n_training_steps_per_head_update

    @partial(jax.jit, static_argnames="self")
    def random_head(self, key: jax.random.PRNGKeyArray, head_probability: jnp.ndarray) -> jnp.int8:
        return jax.random.choice(key, jnp.arange(self.n_heads), p=head_probability)

    def update_target_params(self, step: int) -> None:
        if (step % self.n_training_steps_per_target_update == 0) or (step % self.n_training_steps_per_head_update == 0):
            self.target_params = self.params


class iDQN(BaseMultiHeadQ):
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
        n_training_steps_per_head_update: int,
    ) -> None:
        super().__init__(
            n_heads,
            {"state": jnp.zeros(state_shape, dtype=jnp.float32)},
            n_actions,
            gamma,
            network,
            network_key,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
            n_training_steps_per_head_update,
        )
        self.head_behaviorial_probability = head_behaviorial_probability

    @partial(jax.jit, static_argnames="self")
    def apply(self, params: FrozenDict, states: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply(params, states)

    @partial(jax.jit, static_argnames="self")
    def apply_other_heads(self, params: FrozenDict, states: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply_other_heads(params, states)

    @partial(jax.jit, static_argnames="self")
    def best_action_from_head(
        self, torso_params: FrozenDict, head_params: FrozenDict, state: jnp.ndarray
    ) -> jnp.ndarray:
        """This function is supposed to take a single state and not a batch"""
        return jnp.argmax(self.network.apply_specific_head(torso_params, head_params, state)[0]).astype(jnp.int8)

    @partial(jax.jit, static_argnames="self")
    def compute_target(self, params: FrozenDict, samples: FrozenDict) -> jnp.ndarray:
        # mapping over the states
        return jax.vmap(
            lambda reward, absorbing, max_next_states: reward + (1 - absorbing) * self.gamma * max_next_states,
        )(samples["reward"], samples["absorbing"], jnp.max(self.apply(params, samples["next_state"]), axis=2))

    @partial(jax.jit, static_argnames="self")
    def loss(self, params: FrozenDict, params_target: FrozenDict, samples: FrozenDict) -> jnp.float32:
        targets = self.compute_target(params_target, samples)[:, :-1]
        values_actions = self.apply_other_heads(params, samples["state"])
        # mapping over the states
        predictions = jax.vmap(lambda value_actions, action: value_actions[:, action])(
            values_actions, samples["action"]
        )

        return self.metric(predictions - targets, ord="2")

    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.int8:
        idx_head = self.random_head(key, self.head_behaviorial_probability)

        return self.best_action_from_head(
            params[f"torso_params_{0 if idx_head == 0 else 1}"], params[f"head_params_{idx_head}"], state
        )

    @partial(jax.jit, static_argnames="self")
    def update_heads(self, params: FrozenDict) -> FrozenDict:
        return self.network.update_heads(params)

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer, key: jax.random.PRNGKeyArray) -> jnp.float32:
        loss = super().update_online_params(step, replay_buffer, key)

        if step % self.n_training_steps_per_head_update == 0:
            self.params = self.update_heads(self.params)

        return loss

    def compute_standard_deviation_head(self, replay_buffer: ReplayBuffer, key: jax.random.PRNGKeyArray) -> jnp.float32:
        standard_deviation = 0

        for _ in range(100):
            batch_samples = replay_buffer.sample_random_batch(key)
            standard_deviation += jnp.std(self(self.params, batch_samples["state"]), axis=1).sum()

        return standard_deviation / (100 * replay_buffer.batch_size * self.n_actions)

    def compute_approximation_error(self, replay_buffer: ReplayBuffer, key: jax.random.PRNGKeyArray) -> jnp.float32:
        approximation_error = 0

        for _ in range(500):
            batch_samples = replay_buffer.sample_random_batch(key)

            targets = self.compute_target(self.target_params, batch_samples)[:, 0]
            values_actions = self.apply(self.params, batch_samples["state"])
            # mapping over the states
            predictions = jax.vmap(lambda value_actions, action: value_actions[1, action])(
                values_actions, batch_samples["action"]
            )
            approximation_error += self.metric(predictions - targets, ord="sum")

        return approximation_error / (500 * replay_buffer.batch_size)


class iIQN(BaseMultiHeadQ):
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
        n_training_steps_per_head_update: int,
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
            n_training_steps_per_head_update,
        )
        self.head_behaviorial_probability = head_behaviorial_probability
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
        return self.network.apply_other_heads(params, states, key, self.n_quantiles)

    @partial(jax.jit, static_argnames="self")
    def apply_n_quantiles_target(
        self, params: FrozenDict, states: jnp.ndarray, key: jax.random.PRNGKeyArray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.network.apply(params, states, key, self.n_quantiles_target)

    @partial(jax.jit, static_argnames="self")
    def best_action_from_head(
        self,
        torso_params: FrozenDict,
        quantiles_params: FrozenDict,
        head_params: FrozenDict,
        state: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """This function is supposed to take a single state and not a batch"""
        return jnp.argmax(
            jnp.mean(
                self.network.apply_specific_head(
                    torso_params, quantiles_params, head_params, state, key, self.n_quantiles_policy
                )[0],
                axis=0,
            )
        ).astype(jnp.int8)

    def add_keys(self, samples):
        self.network_key, samples["key"], samples["next_key"], samples["policy_key"] = jax.random.split(
            self.network_key, 4
        )

    @partial(jax.jit, static_argnames="self")
    def compute_target(self, params: FrozenDict, samples: FrozenDict) -> jnp.ndarray:
        next_states_policy_quantiles_actions, _ = self.apply_n_quantiles_policy(
            params, samples["next_state"], samples["policy_key"]
        )  # output (batch_size, n_heads, n_quantiles_policy, n_actions)
        next_states_policy_values_actions = jnp.mean(
            next_states_policy_quantiles_actions, axis=2
        )  # output (batch_size, n_heads, n_actions)
        next_states_action = jnp.argmax(next_states_policy_values_actions, axis=2)  # output (batch_size, n_heads)

        next_states_quantiles_actions, _ = self.apply_n_quantiles_target(
            params, samples["next_state"], samples["next_key"]
        )  # output (batch_size, n_heads, n_quantiles_target, n_actions)

        # mapping first over the states and then over the heads
        next_states_quantiles = jax.vmap(jax.vmap(lambda quantiles_actions, action: quantiles_actions[:, action]))(
            next_states_quantiles_actions, next_states_action
        )  # output (batch_size, n_heads, n_quantiles_target)

        # mapping over the states
        return jax.vmap(
            lambda reward, absorbing, next_states_quantiles_: reward
            + (1 - absorbing) * self.gamma * next_states_quantiles_,
        )(
            samples["reward"], samples["absorbing"], next_states_quantiles
        )  # output (batch_size, n_heads, n_quantiles_target)

    @partial(jax.jit, static_argnames="self")
    def loss(self, params: FrozenDict, params_target: FrozenDict, samples: FrozenDict) -> jnp.float32:
        targets = self.compute_target(params_target, samples)  # output (batch_size, n_heads, n_quantiles_target)

        states_quantiles_actions, quantiles = self.apply_n_quantiles(
            params, samples["state"], samples["key"]
        )  # output (batch_size, n_heads - 1, n_quantiles, n_actions) | (batch_size, n_quantiles)
        # mapping over the states
        predictions = jax.vmap(lambda quantiles_actions, action: quantiles_actions[:, :, action])(
            states_quantiles_actions, samples["action"]
        )  # output (batch_size, n_heads - 1, n_quantiles)

        # cross difference
        bellman_errors = (
            targets[:, :-1, :, jnp.newaxis] - predictions[:, :, jnp.newaxis]
        )  # output (batch_size, n_heads - 1, n_quantiles_target, n_quantiles)

        huber_losses_quadratic_case = (jnp.abs(bellman_errors) <= 1).astype(jnp.float32) * 0.5 * bellman_errors**2
        huber_losses_linear_case = (jnp.abs(bellman_errors) > 1).astype(jnp.float32) * (jnp.abs(bellman_errors) - 0.5)
        huber_losses = (
            huber_losses_quadratic_case + huber_losses_linear_case
        )  # output (batch_size, n_heads - 1, n_quantiles_target, n_quantiles)

        # mapping first over the heads and then over the target quantiles
        quantile_losses = jax.vmap(
            jax.vmap(
                lambda quantile, bellman_errors_, huber_losses_: jnp.abs(
                    quantile - jax.lax.stop_gradient(bellman_errors_ < 0).astype(jnp.float32)
                )
                * huber_losses_,
                (None, 1, 1),
                1,
            ),
            (None, 1, 1),
            1,
        )(
            quantiles, bellman_errors, huber_losses
        )  # output (batch_size, n_heads - 1, n_quantiles_target, n_quantiles)

        # sum over the quantiles and mean over the target quantiles, the heads and the states
        return jnp.mean(jnp.sum(quantile_losses, axis=3))

    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.int8:
        idx_head = self.random_head(key, self.head_behaviorial_probability)

        return self.best_action_from_head(
            params["torso_params_0" if idx_head == 0 else "torso_params_1"],
            params["quantiles_params_0" if idx_head == 0 else "quantiles_params_1"],
            params[f"head_params_{idx_head}"],
            state,
            key,
        )

    @partial(jax.jit, static_argnames="self")
    def update_heads(self, params: FrozenDict) -> FrozenDict:
        return self.network.update_heads(params)

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer, key: jax.random.PRNGKeyArray) -> jnp.float32:
        loss = super().update_online_params(step, replay_buffer, key)

        if step % self.n_training_steps_per_head_update == 0:
            self.params = self.update_heads(self.params)

        return loss
