from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slim_idqn.networks.architectures.shared_layer_DQN import SharedLayeriDQNNet
from slim_idqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class SharedLayeriDQN:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        features: list,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_frequency: int,
        shift_params_frequency: int,
        learning_rate: float,
        num_shared_layers=1,
        adam_eps: float = 1e-8,
        num_networks: int = 5,
    ):
        self.num_networks = num_networks
        self.network = SharedLayeriDQNNet(num_actions=n_actions, num_shared_layers=num_shared_layers, num_heads= num_networks, features=features, observation_dim=observation_dim)
        self.online_params = self.network.init(key)
        
        self.target_params = self.online_params

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.online_params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.learning_rate = learning_rate
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
    
    def shift_params(self, step: int):
        if step % self.shift_params_frequency == 0:
            self.online_params = FrozenDict(
                head_params=self.network.roll(self.online_params["head_params"]),
                torso_params=self.online_params["torso_params"]
            )
            self.target_params = self.online_params
            logs = {"loss": self.cumulated_loss / (self.shift_params_frequency / self.update_to_data)}
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
        loss, grad_losses = jax.value_and_grad(self.loss_on_batch)(params, params_target, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_losses, optimizer_state)  
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    def loss_on_batch(self, params: FrozenDict, params_target: FrozenDict, samples):
        losses = jax.vmap(self.loss, in_axes=(None, None, 0))(params, params_target, samples)

        return losses.mean()

    def loss(self, params: FrozenDict, params_target: FrozenDict, sample: ReplayElement):
        # computes the loss for a single sample
        target = self.compute_target(params_target, sample)
        q_value = self.network.apply_other_heads(params, sample.state.squeeze(1))[:, sample.action]

        return jnp.square(q_value - target)

    def compute_target(self, params, sample: ReplayElement):
        max_next_q = jnp.max(self.network.apply(params, sample.next_state.squeeze(1))[:-1], axis=1)

        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * max_next_q

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, network_selection_key):
        head_idx = jax.random.randint(network_selection_key, (), 0, self.num_networks)
        # computes the best action for a single state and head
        return jnp.argmax(self.network.apply_specific_head(params,head_idx, state))

    def get_model(self):
        return {"params": self.online_params}
