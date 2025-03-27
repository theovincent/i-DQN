from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


@jax.jit
def shift_params(params):
    # Each online network is updated to the following online network
    # \theta_k <- \theta_{k + 1}, i.e., params[k] <- params[k + 1]
    return jax.tree.map(lambda param: param.at[:-1].set(param[1:]), params)


@jax.jit
def sync_target_params(params, target_params):
    # Each target network is synchronized to the online network it represents
    # \bar{\theta}_k <- \theta_k, i.e., target_params[k] <- params[k-1]
    return jax.tree.map(lambda param, target_param: target_param.at[1:].set(param[:-1]), params, target_params)


class iDQN:
    def __init__(
        self,
        key: jax.Array,
        observation_dim,
        n_actions,
        n_networks: int,
        features: list,
        architecture_type: str,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_frequency: int,
        target_sync_frequency: int,
        adam_eps: float = 1e-8,
    ):
        self.n_networks = n_networks
        self.network = DQNNet(features, architecture_type, n_actions)
        # Create K online parameters
        # params = [\theta_1, \theta_2, ..., \theta_K]
        self.params = jax.vmap(self.network.init, in_axes=(0, None))(
            jax.random.split(key, self.n_networks), jnp.zeros(observation_dim, dtype=jnp.float32)
        )

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = jax.vmap(self.optimizer.init)(self.params)
        # Create K target parameters
        # target_params = [\bar{\theta}_0, \bar{\theta}_1, ..., \bar{\theta}_{K-1}]
        self.target_params = self.params

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        self.target_sync_frequency = target_sync_frequency
        self.cumulated_losses = np.zeros(self.n_networks)

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.update_to_data == 0:
            batch_samples = replay_buffer.sample()

            self.params, self.optimizer_state, losses = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples
            )
            self.cumulated_losses += losses

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            # Each target network is updated to its respective online network
            # \bar{\theta}_k <- \theta_{k + 1}, i.e., target_params[k] <- params[k]
            self.target_params = self.params.copy()
            # Window shift
            self.params = shift_params(self.params)

            logs = {"loss": np.mean(self.cumulated_losses) / (self.target_update_frequency / self.update_to_data)}
            for idx_network in range(self.n_networks):
                logs[f"networks/{idx_network}_loss"] = self.cumulated_losses[idx_network] / (
                    self.target_update_frequency / self.update_to_data
                )
            self.cumulated_losses = np.zeros_like(self.cumulated_losses)

            return True, logs

        if step % self.target_sync_frequency == 0:
            self.target_params = sync_target_params(self.params, self.target_params)

        return False, {}

    @partial(jax.jit, static_argnames="self")
    @partial(jax.vmap, in_axes=(None, 0, 0, 0, None))  # vmap over the Q-networks
    def learn_on_batch(
        self,
        params: FrozenDict,
        params_target: FrozenDict,
        optimizer_state,
        batch_samples,
    ):
        loss, grad_loss = jax.value_and_grad(self.loss_on_batch)(params, params_target, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    def loss_on_batch(self, params: FrozenDict, params_target: FrozenDict, samples):
        return jax.vmap(self.loss, in_axes=(None, None, 0))(params, params_target, samples).mean()

    def loss(self, params: FrozenDict, params_target: FrozenDict, sample: ReplayElement):
        # computes the loss for a single sample
        target = self.compute_target(params_target, sample)
        q_value = self.network.apply(params, sample.state)[sample.action]
        return jnp.square(q_value - target)

    def compute_target(self, params: FrozenDict, sample: ReplayElement):
        # computes the target value for single sample
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            self.network.apply(params, sample.next_state)
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.Array):
        idx_params = jax.random.randint(key, (), 0, self.n_networks)

        # computes the best action for a single state
        return jnp.argmax(self.network.apply(jax.tree.map(lambda param: param[idx_params], params), state))

    def get_model(self):
        return {"params": self.params}
