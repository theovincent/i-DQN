from typing import Sequence

import flax.linen as nn
import jax
from flax.core import FrozenDict
import jax.numpy as jnp

class Torso(nn.Module):
    features: Sequence[int]
    
    @nn.compact
    def __call__(self, x):
        initializer = nn.lecun_normal()
        for layer_size in self.features:
            x = nn.relu(nn.Dense(layer_size, kernel_init=initializer)(x))
        return x

class Head(nn.Module):
    features: Sequence[int]
    n_actions: int
    @nn.compact
    def __call__(self, x):
        initializer = nn.lecun_normal()
        for layer_size in self.features:
            x = nn.relu(nn.Dense(layer_size, kernel_init=initializer)(x))
        
        return nn.Dense(self.n_actions, kernel_init = initializer)


class SharedLayerDQN:
    def __init__(
        self,        
        features: list,
        num_actions,
        num_shared_layers=1,
        num_heads=5,
        observation_dim: int
    ):
        self.num_actions = num_actions
        self.num_heads = num_heads
        self.num_shared_layers = num_shared_layers
        self.observation_dim = observation_dim
        
        self.torso = Torso(features = features[:num_shared_layers])
        self.head = Head(features=features[num_shared_layers:], n_actions=self.num_actions)

    def init_torso(self, key):

        return self.torso.init(key, jnp.zeros(self.observation_dim, dtype=jnp.float32))

    def init_heads(self, keys):

        return  jax.vmap(self.head.init, in_axes=(0, None))(keys, jnp.zeros(self.observation_dim, dtype=jnp.float32))

    def init(self, key):
        torso_key, *head_keys = jax.random.split(key, self.num_heads + 2)
        torso_params = self.init_torso(torso_key)
        head_params =  self.init_heads(head_keys)

        return FrozenDict(torso_params=torso_params, head_params=head_params)
  

    def apply(self, params, state):
        features = self.torso.apply(params["torso_params"], state)

        return jax.vmap(self.head.apply, in_axes=(0, None))(params["head_params"], features)

    
    def roll(self, params):
        return jax.tree_util.tree_map(lambda param: param.at[:-1].set(param[1:]), params)
    
 
    
    def apply_other_heads(self, params, state):
        features = self.torso.apply(params["torso_params"], state)
        remaining_head_params = jax.tree_util.tree_map(lambda param: param[1:], params["head_params"])

        return jax.vmap(self.head.apply, in_axes=(0, None))(remaining_head_params, features)

