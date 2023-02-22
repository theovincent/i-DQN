from typing import Dict
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp


class ReplayBuffer:
    def __init__(self, max_size: int) -> None:
        self.max_size: int = max_size
        self.len: int = 0
        self.idx: int = 0

    def set_first(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        absorbing: np.ndarray,
    ) -> None:
        self.states = np.zeros((self.max_size,) + state.shape, dtype=state.dtype)
        self.actions = np.zeros(self.max_size, dtype=action.dtype)
        self.rewards = np.zeros(self.max_size, dtype=reward.dtype)
        self.next_states = np.zeros((self.max_size,) + next_state.shape, dtype=next_state.dtype)
        self.absorbings = np.zeros(self.max_size, dtype=absorbing.dtype)

    def add_sample(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        absorbing: np.ndarray,
        idx: int,
    ) -> None:
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.absorbings[idx] = absorbing

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        absorbing: np.ndarray,
    ) -> None:
        if self.len == 0:
            self.set_first(state, action, reward, next_state, absorbing)
        self.add_sample(state, action, reward, next_state, absorbing, self.idx)

        self.idx += 1
        self.len = min(self.len + 1, self.max_size)
        if self.idx >= self.max_size:
            self.idx = 0

    def save(self, path: str) -> None:
        np.savez(
            path,
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            next_states=self.next_states,
            absorbings=self.absorbings,
        )

    def load(self, path: str) -> None:
        dataset = np.load(path)

        self.states = np.array(dataset["states"])
        self.actions = np.array(dataset["actions"])
        self.rewards = np.array(dataset["rewards"])
        self.next_states = np.array(dataset["next_states"])
        self.absorbings = np.array(dataset["absorbings"])

        self.len = self.states.shape[0]

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
