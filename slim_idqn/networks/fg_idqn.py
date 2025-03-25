from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slim_idqn.networks.architectures.dqn import DQNNet
from slim_idqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement
from slim_idqn.networks.idqn import iDQN


class FG_iDQN:
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
        shift_params_frequency: int,
        adam_eps: float = 1e-8,
        num_networks: int = 5
    ):
        keys = jax.random.split(key, num=num_networks)
        self.num_networks = num_networks

        self.network = DQNNet(features, architecture_type, n_actions)
        self.online_params= jax.vmap(self.network.init, in_axes=(0, None))(keys, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.online_params)
                                            

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.shift_params_frequency = shift_params_frequency
        self.cumulated_loss = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.update_to_data == 0:
            batch_samples = replay_buffer.sample()

            self.online_params, self.optimizer_states, loss = self.single_learn_on_batch(
                self.online_params,
                self.optimizer_state,
                batch_samples
            )
            self.cumulated_loss += loss

    def loss_on_batch(self, online_params, samples):
        return jax.vmap(self.loss, in_axes=(None, 0))(online_params, samples).mean()
    
    def loss(self, online_params, sample: ReplayElement):
        # computes the loss for a single sample
        target = self.compute_target(online_params, sample)
        q_value = self.apply_other_heads(online_params, sample.state)[:,sample.action]

        return jnp.square(q_value - target)
    
    
    def compute_target(self, online_params, sample: ReplayElement):
        max_next_q = jnp.max(jax.vmap(self.network.apply, in_axes=(0, None))(online_params, sample.next_state)[:-1], axis=1)

        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * max_next_q
    
    def shift_params(self, step: int):
        if step % self.shift_params_frequency == 0:
            self.online_params = self.roll(self.online_params)

            logs = {"loss": self.cumulated_loss / (self.shift_params_frequency / self.update_to_data)}
            self.cumulated_loss = 0

            return True, logs
        return False, {}

    def apply_other_heads(self, params, state):
        remaining_params = jax.tree_util.tree_map(lambda param: param[1:], params)

        return jax.vmap(self.network.apply, in_axes=(0, None))(remaining_params, state)

    

    @partial(jax.jit, static_argnames="self")
    def single_learn_on_batch(
        self,
        online_params,
        optimizer_state,
        batch_samples,
    ):
        loss, grad_loss = jax.value_and_grad(self.loss_on_batch)(online_params, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(online_params, updates)

        return params, optimizer_state, loss

    def roll(self, params):
        return jax.tree_util.tree_map(lambda param: param.at[:-1].set(param[1:]), params)
    
    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, network_selection_key):
        head_idx = jax.random.randint(network_selection_key, (), 0, self.num_networks)
        sampled_params = jax.tree_util.tree_map(lambda param: param[head_idx], params)
        # computes the best action for a single state and head
        return jnp.argmax(self.network.apply(sampled_params, state))


    def get_model(self):
        return {"params": self.online_params}
