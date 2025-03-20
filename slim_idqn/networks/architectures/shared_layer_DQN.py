from typing import Sequence

import flax.linen as nn
import jax
from flax.core import FrozenDict
import jax.numpy as jnp

class Torso(nn.Module):
    features: Sequence[int]
    
    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.lecun_normal()
        for layer_size in self.features:
            x = nn.relu(nn.Dense(layer_size, kernel_init=initializer)(x))
        return x

class Head(nn.Module):
    features: Sequence[int]
    n_actions: int
    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.lecun_normal()
        for layer_size in self.features:
            x = nn.relu(nn.Dense(layer_size, kernel_init=initializer)(x))
        
        return nn.Dense(self.n_actions, kernel_init = initializer)(x)


class SharedLayeriDQNNet:
    def __init__(
        self,        
        observation_dim: int,
        features: list,
        num_actions,
        num_heads=5,
        num_shared_layers=1,
    ):
        self.num_actions = num_actions
        self.num_heads = num_heads + 1
        self.num_shared_layers = num_shared_layers
        self.observation_dim = observation_dim
        self.features = features
        
        self.torso = Torso(features = features[:num_shared_layers])
        self.head = Head(features=features[num_shared_layers:], n_actions=self.num_actions)

    def init_torso(self, key, state):

        return self.torso.init(key, state)

    def init_heads(self, key, state):
        head_keys = jax.random.split(key, self.num_heads)
        return  jax.vmap(self.head.init, in_axes=(0, None))(head_keys, state)

    def init(self, key):
        dummy_input = jnp.zeros(self.observation_dim, dtype=jnp.float32) 
        torso_key, head_key = jax.random.split(key, 2)
        torso_params = self.init_torso(torso_key, dummy_input)
        
        dummy_features = self.torso.apply(torso_params, dummy_input)
        head_params =  self.init_heads(head_key, dummy_features)

        return FrozenDict(torso_params=torso_params, head_params=head_params)
  

    def apply(self, params, state):
        features = self.torso.apply(params["torso_params"], state)

        return jax.vmap(self.head.apply, in_axes=(0, None))(params["head_params"], features)

    
    def roll(self, params):
        return jax.tree_util.tree_map(lambda param: param.at[:-1].set(param[1:]), params)
    

    def apply_specific_head(self, params, head_idx, state):
        features = self.torso.apply(params["torso_params"], state)
        sample_network_head_params = jax.tree_util.tree_map(lambda param: param[head_idx], params["head_params"])
        return self.head.apply(sample_network_head_params, features)

    
    def apply_other_heads(self, params, state):
        features = self.torso.apply(params["torso_params"], state)
        remaining_head_params = jax.tree_util.tree_map(lambda param: param[1:], params["head_params"])

        return jax.vmap(self.head.apply, in_axes=(0, None))(remaining_head_params, features)

