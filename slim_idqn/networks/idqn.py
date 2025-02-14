from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slim_idqn.networks.architectures.dqn import DQNNet
from slim_idqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


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
        keys = jax.random.split(key, num=num_networks+1)
        self.num_networks = num_networks
        self.network = DQNNet(features, architecture_type, n_actions)

        self.online_params_params = jax.vmap(self.network.init, in_axes=(0, None))(keys, jnp.zeros(observation_dim, dtype=jnp.float32))
        self.target_params = self.online_params

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.network.init(keys[0], jnp.zeros(observation_dim, dtype=jnp.float32)))

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        self.shift_params_frequency = shift_params_frequency
        self.cumulated_loss = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.update_to_data == 0:
            batch_samples = replay_buffer.sample()

            self.online_params, self.optimizer_state, loss = self.learn_on_batch(
                self.online_params,
                self.target_params,
                self.optimizer_state,
                batch_samples
            )
            self.cumulated_loss += loss


    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            self.target_params = self.online_params

            logs = {"loss": self.cumulated_loss / (self.target_update_frequency / self.update_to_data)}
            self.cumulated_loss = 0

            return True, logs
        return False, {}
    
    def roll(self, params):
        return jax.tree_util.tree_map(lambda param: param.at[:-1].set(param[1:]), params)
    
    def shift_params(self, step: int):
        if step % self.shift_params_frequency == 0:
            self.target_params = self.roll(self.online_params)

            logs = {"loss": self.cumulated_loss / (self.target_update_frequency / self.update_to_data)}
            self.cumulated_loss = 0

            return True, logs
        return False, {}
    

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self,
        params: FrozenDict,
        params_target: FrozenDict,
        optimizer_state,
        batch_samples,
    ):
        losses, grad_losses = jax.value_and_grad(self.loss_on_batch)(params, params_target, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_losses, optimizer_state)  
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, losses

    def loss_on_batch(self, params: FrozenDict, params_target: FrozenDict, samples):
        losses =  jax.vmap(self.loss, in_axes=(None, None, 0))(params, params_target, samples)

        return losses.mean()

    def loss(self, params: FrozenDict, params_target: FrozenDict, sample: ReplayElement):
        # computes the loss for a single sample
        target = self.compute_target(params_target, sample)
        q_value = self.apply_other_heads(params, sample.state)

        return jnp.square(q_value - target)
    
    def apply_other_heads(self, params, state):
        remaining_params = jax.tree_util.tree_map(lambda param: param[1:], params)

        return jax.vmap(self.network.apply, in_axis=(0, None))(remaining_params, state)

    def compute_target(self, params, sample: ReplayElement):
        max_next_q = jnp.max(jax.vmap(self.network.apply, in_axis=(0, None))(params, sample.next_state)[:-1], axis=0)

        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * max_next_q

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        # computes the best action for a single state
        return jnp.argmax(self.network.apply(params, state))

    def get_model(self):
        return {"params": self.online_params}
