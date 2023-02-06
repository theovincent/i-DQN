from typing import Dict
import numpy as np
import jax.numpy as jnp
import jax


class ReplayBuffer:
    def __init__(self, max_size: int) -> None:
        self.max_size: int = max_size
        self.len: int = 0
        self.idx: int = 0

    def set_first(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        reward: jnp.ndarray,
        next_state: jnp.ndarray,
        absorbing: jnp.ndarray,
    ) -> None:
        self.states = jnp.zeros((self.max_size,) + state.shape, dtype=state.dtype)
        self.actions = jnp.zeros(self.max_size, dtype=action.dtype)
        self.rewards = jnp.zeros(self.max_size, dtype=reward.dtype)
        self.next_states = jnp.zeros((self.max_size,) + next_state.shape, dtype=next_state.dtype)
        self.absorbings = jnp.zeros(self.max_size, dtype=absorbing.dtype)

    def add_sample(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        reward: jnp.ndarray,
        next_state: jnp.ndarray,
        absorbing: jnp.ndarray,
        idx: int,
    ) -> None:
        self.states = self.states.at[idx].set(state)
        self.actions = self.actions.at[idx].set(action)
        self.rewards = self.rewards.at[idx].set(reward)
        self.next_states = self.next_states.at[idx].set(next_state)
        self.absorbings = self.absorbings.at[idx].set(absorbing)

    def add(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        reward: jnp.ndarray,
        next_state: jnp.ndarray,
        absorbing: jnp.ndarray,
    ) -> None:
        if self.idx >= self.max_size:
            self.idx = 0

        if self.len == 0:
            self.set_first(state, action, reward, next_state, absorbing)
        self.add_sample(state, action, reward, next_state, absorbing, self.idx)

        self.idx += 1
        self.len = min(self.len + 1, self.max_size)

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

        self.states = jnp.array(dataset["states"])
        self.actions = jnp.array(dataset["actions"])
        self.rewards = jnp.array(dataset["rewards"])
        self.next_states = jnp.array(dataset["next_states"])
        self.absorbings = jnp.array(dataset["absorbings"])

        self.len = self.states.shape[0]

    def sample_random_batch(self, sample_key: jax.random.PRNGKeyArray, n_samples: int) -> Dict[str, jnp.ndarray]:
        idxs = jax.random.randint(sample_key, shape=(n_samples,), minval=0, maxval=self.len)

        return {
            "state": self.states[idxs],
            "action": self.actions[idxs],
            "reward": self.rewards[idxs],
            "next_state": self.next_states[idxs],
            "absorbing": self.absorbings[idxs],
        }
