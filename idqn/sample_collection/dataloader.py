import numpy as np
import jax
import jax.numpy as jnp

from idqn.sample_collection.replay_buffer import ReplayBuffer


class SampleDataLoader:
    def __init__(self, replay_buffer: ReplayBuffer, batch_size: int, shuffle_key: jax.random.PRNGKeyArray) -> None:
        self.replay_buffer = replay_buffer
        self.n_samples = replay_buffer.len
        self.batch_size = batch_size
        self.shuffle_key = shuffle_key

        self.indexes = jnp.arange(0, self.n_samples)

        self.jitted_getitem = jax.jit(self.getitem)

    def __len__(self) -> int:
        return np.ceil(self.n_samples / self.batch_size).astype(int)

    def getitem(self, idxs) -> dict:
        return {
            "state": self.replay_buffer.states[idxs],
            "action": self.replay_buffer.actions[idxs],
            "reward": self.replay_buffer.rewards[idxs],
            "next_state": self.replay_buffer.next_states[idxs],
            "absorbing": self.replay_buffer.absorbings[idxs],
        }

    def __getitem__(self, idx: int) -> dict:
        assert 0 <= idx and idx <= len(self), f"The queried index {idx} is out of scope [0, {len(self)}]."
        if idx == len(self):
            raise StopIteration

        start = idx * self.batch_size
        end = jnp.minimum((idx + 1) * self.batch_size, self.n_samples)
        idxs = self.indexes[start:end]

        return self.jitted_getitem(idxs)

    def shuffle(self) -> None:
        self.shuffle_key, key = jax.random.split(self.shuffle_key)
        self.indexes = jax.random.permutation(key, self.indexes)
