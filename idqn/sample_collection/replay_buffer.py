import shutil
from typing import Dict, Type
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp


class ReplayBuffer:
    def __init__(
        self, max_size: int, path: str, state_shape: list, state_dtype: Type, reward_dtype: Type, overwrite: bool
    ) -> None:
        self.max_size = max_size
        self.path = path

        self.state_shape = state_shape
        self.state_dtype = state_dtype
        self.action_dtype = np.int8
        self.reward_dtype = reward_dtype
        self.absorbing_dtype = np.bool_

        self.load(overwrite, counters_loaded=False)

    def load(self, overwrite: bool, counters_loaded: bool) -> None:
        if overwrite:
            mode = "w+"
            self.len = 0
            self.idx = 0
        else:
            mode = "r+"
            if not counters_loaded:
                self.len = np.load(self.path + "_len.npy")
                self.idx = np.load(self.path + "_idx.npy")

        self.states = np.memmap(
            self.path + "_states", mode=mode, shape=(self.max_size,) + self.state_shape, dtype=self.state_dtype
        )
        self.actions = np.memmap(self.path + "_actions", mode=mode, shape=self.max_size, dtype=self.action_dtype)
        self.rewards = np.memmap(self.path + "_rewards", mode=mode, shape=self.max_size, dtype=self.reward_dtype)
        self.next_states = np.memmap(
            self.path + "_next_states", mode=mode, shape=(self.max_size,) + self.state_shape, dtype=self.state_dtype
        )
        self.absorbings = np.memmap(
            self.path + "_absorbings", mode=mode, shape=self.max_size, dtype=self.absorbing_dtype
        )

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        absorbing: np.ndarray,
    ) -> None:
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.absorbings[self.idx] = absorbing

        self.idx += 1
        self.len = min(self.len + 1, self.max_size)
        if self.idx >= self.max_size:
            self.idx = 0

    def save(self, new_path: str) -> None:
        np.save(self.path + "_len", self.len)
        np.save(self.path + "_idx", self.idx)

        self.states.flush()
        self.actions.flush()
        self.rewards.flush()
        self.next_states.flush()
        self.absorbings.flush()

        shutil.copy(self.path + "_states", new_path + "_states")
        shutil.copy(self.path + "_actions", new_path + "_actions")
        shutil.copy(self.path + "_rewards", new_path + "_rewards")
        shutil.copy(self.path + "_next_states", new_path + "_next_states")
        shutil.copy(self.path + "_absorbings", new_path + "_absorbings")

        self.path = new_path
        self.load(overwrite=False, counters_loaded=True)

    def sample_random_batch(self, sample_key: jax.random.PRNGKeyArray, n_samples: int) -> Dict[str, jnp.ndarray]:
        idxs = self.get_sample_indexes(sample_key, n_samples, self.len)
        return self.create_batch(
            self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.absorbings[idxs]
        )

    @staticmethod
    @partial(jax.jit, static_argnames="n_samples")
    def get_sample_indexes(key: jax.random.PRNGKeyArray, n_samples: int, maxval: int) -> jnp.ndarray:
        return jax.random.randint(key, shape=(n_samples,), minval=0, maxval=maxval)

    @staticmethod
    @jax.jit
    def create_batch(
        states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray, absorbings: np.ndarray
    ) -> Dict[str, jnp.ndarray]:
        return {
            "state": jnp.array(states, dtype=jnp.float32),
            "action": jnp.array(actions, dtype=jnp.int8),
            "reward": jnp.array(rewards, dtype=jnp.float32),
            "next_state": jnp.array(next_states, dtype=jnp.float32),
            "absorbing": jnp.array(absorbings, dtype=jnp.bool_),
        }
