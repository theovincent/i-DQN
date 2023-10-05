from time import time
import jax
import numpy as np

from idqn.sample_collection.replay_buffer import ReplayBuffer


RANDOM_SEED = np.random.randint(1000)


def run_cli():
    print("Time Replay Buffer")
    time_atari_q = TimeReplayBuffer()

    time_atari_q.time_add()
    time_atari_q.time_sample_batch()


class TimeReplayBuffer:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = RANDOM_SEED
        print(f"random seed {self.random_seed}")
        self.n_runs = 30
        self.key = jax.random.PRNGKey(self.random_seed)
        self.observation_shape = (84, 84)
        self.replay_capacity = jax.random.randint(self.key, (), minval=50, maxval=1000)
        self.batch_size = jax.random.randint(self.key, (), minval=1, maxval=self.replay_capacity)
        self.update_horizon = 1
        self.gamma = jax.random.uniform(self.key)
        self.identity = lambda x: x

    def time_add(self) -> None:
        replay_buffer = ReplayBuffer(
            self.observation_shape,
            self.replay_capacity,
            self.batch_size,
            self.update_horizon,
            self.gamma,
            self.identity,
        )
        batch_key = self.key

        t_begin = time()

        for _ in range(self.n_runs):
            batch_key, key = jax.random.split(batch_key)
            observation = np.array(
                jax.random.randint(key, replay_buffer._observation_shape, 0, 256, replay_buffer._observation_dtype)
            )
            action = np.array(jax.random.randint(key, (), 0, 10, replay_buffer._action_dtype))
            reward = np.array(jax.random.uniform(key, (), replay_buffer._reward_dtype, 0, 1000))
            terminal = np.array(jax.random.bernoulli(key, 0.01, ()), dtype=replay_buffer._terminal_dtype)
            batch_key, key_truncateds = jax.random.split(self.key)
            truncated = np.array(jax.random.bernoulli(key_truncateds, 0.01, ()), dtype=replay_buffer._terminal_dtype)

            replay_buffer.add(observation, action, reward, terminal, episode_end=terminal or truncated)

        print("Time add: ", (time() - t_begin) / self.n_runs)

    def time_sample_batch(self) -> None:
        replay_buffer = ReplayBuffer(
            self.observation_shape,
            self.replay_capacity,
            self.batch_size,
            self.update_horizon,
            self.gamma,
            self.identity,
        )
        batch_key = self.key

        for _ in range(10 * self.batch_size):
            batch_key, key = jax.random.split(batch_key)
            observation = np.array(
                jax.random.randint(key, replay_buffer._observation_shape, 0, 256, replay_buffer._observation_dtype)
            )
            action = np.array(jax.random.randint(key, (), 0, 10, replay_buffer._action_dtype))
            reward = np.array(jax.random.uniform(key, (), replay_buffer._reward_dtype, 0, 1000))
            terminal = np.array(jax.random.bernoulli(key, 0.01, ()), dtype=replay_buffer._terminal_dtype)
            batch_key, key_truncateds = jax.random.split(self.key)
            truncated = np.array(jax.random.bernoulli(key_truncateds, 0.01, ()), dtype=replay_buffer._terminal_dtype)

            replay_buffer.add(observation, action, reward, terminal, episode_end=terminal or truncated)

        t_begin = time()

        for _ in range(self.n_runs):
            replay_buffer.sample_transition_batch()

        print("Time sample batch: ", (time() - t_begin) / self.n_runs)
