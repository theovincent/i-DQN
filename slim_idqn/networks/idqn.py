from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class iDQN:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        features: list,
        architecture_type: str,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_frequency: int,
        shift_params_frequency: int,
        adam_eps: float = 1e-8,
        num_networks: int = 5
    ):
        first_key, remaining_keys = jax.random.split(key, num=num_networks+1)

        self.num_networks = num_networks
        self.network = DQNNet(features, architecture_type, n_actions)
        self.online_params = jax.vmap(self.network.init, in_axes=(0, None))(remaining_keys, jnp.zeros(observation_dim, dtype=jnp.float32))
        self.target_params = jnp.array(self.network.init(first_key, jnp.zeros(observation_dim, dtype=jnp.float32)))
        self.target_params.append(self.online_params[:-1])

        #for k in range(num_networks):
        #    params = self.network.init(keys[k+1], jnp.zeros(observation_dim, dtype=jnp.float32))
        #    self.online_params.append(params)
        #    self.target_params.append(self.online_params[k-1])
        #self.online_params = jnp.asarray(self.online_params)
        #self.target_params = jnp.asarray(self.target_params)
        self.optimizers = jnp.full(self.num_networks, optax.adam(learning_rate, eps=adam_eps))
        self.optimizer_states = jnp.asarray(list([self.optimizers[k].init(self.online_params[k]) for k in range(self.num_networks)]))

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        self.shift_params_frequency = shift_params_frequency
        self.cumulated_loss = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.update_to_data == 0:
            batch_samples = replay_buffer.sample()

            self.online_params, self.optimizer_states, losses = jax.vmap(self.single_learn_on_batch, in_axes=(0, 0, 0, None))(
                self.online_params,
                self.target_params,
                self.optimizer_states,
                batch_samples
            )
            loss = losses[-1]

            #for k in range(self.num_networks):
            #    self.online_params[k], self.optimizer_state, loss = self.single_learn_on_batch(
            #        self.online_params[k],
            #        self.target_params[k],
            #        self.optimizer_state,
            #        batch_samples
            #    )

            self.cumulated_loss += loss

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            for k in range(1, self.num_networks):
                self.target_params[k] = self.online_params[k-1].copy()
                
            logs = {"loss": self.cumulated_loss / (self.target_update_frequency / self.update_to_data)}
            self.cumulated_loss = 0

            return True, logs
        return False, {}
    
    def shift_params(self, step: int):
        if step % self.shift_params_frequency == 0:
            for k in range(0, self.num_networks):
                self.target_params[k] = self.online_params[k].copy()

            logs = {"loss": self.cumulated_loss / (self.target_update_frequency / self.update_to_data)}
            self.cumulated_loss = 0

            return True, logs
        return False, {}
    



    @partial(jax.jit, static_argnames="self")
    def single_learn_on_batch(
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
    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        # computes the best action for a single state
        return jnp.argmax(self.network.apply(params, state))

    def get_model(self):
        return {"params": self.online_params}
