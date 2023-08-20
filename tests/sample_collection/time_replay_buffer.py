from time import time
import jax
import numpy as np

from idqn.sample_collection.replay_buffer import ReplayBuffer, NStepReplayBuffer


RANDOM_SEED = np.random.randint(1000)


def run_cli():
    print("Time Replay Buffer")
    time_atari_q = TimeReplayBuffer()

    time_atari_q.time_add()
    time_atari_q.time_sample_batch()

    print("\n\nTime n-step Replay Buffer")
    time_atari_q = TimeNStepReplayBuffer()

    time_atari_q.time_add()


class TimeReplayBuffer:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = RANDOM_SEED
        print(f"random seed {self.random_seed}")
        self.n_runs = 3000
        self.key = jax.random.PRNGKey(self.random_seed)
        self.max_size = jax.random.randint(self.key, (), minval=1, maxval=1000)
        self.batch_size = jax.random.randint(self.key, (), minval=1, maxval=self.max_size)
        self.state_shape = (4, 84, 84)
        self.state_dtype = np.uint8
        self.path = "tests/replay_buffer"
        self.identity = lambda x: x

    def time_add(self) -> None:
        replay_buffer = ReplayBuffer(self.max_size, self.batch_size, self.state_shape, self.state_dtype, self.identity)
        batch_key = self.key

        t_begin = time()

        for _ in range(self.n_runs):
            batch_key, key = jax.random.split(batch_key)
            state = np.array(jax.random.randint(key, replay_buffer.state_shape, 0, 256, replay_buffer.state_dtype))
            action = np.array(jax.random.randint(key, (), 0, 10, replay_buffer.action_dtype))
            reward = np.array(jax.random.uniform(key, (), replay_buffer.reward_dtype, 0, 1000))
            absorbing = np.array(jax.random.randint(key, (), 0, 2), dtype=replay_buffer.absorbing_dtype)
            batch_key, key_truncateds, keys_next_states = jax.random.split(self.key, 3)
            truncated = np.array(jax.random.randint(key_truncateds, (), 0, 2), dtype=replay_buffer.absorbing_dtype)
            next_state = np.array(
                jax.random.randint(keys_next_states, replay_buffer.state_shape, 1, 256, replay_buffer.state_dtype)
            )

            replay_buffer.add(state, action, reward, next_state, absorbing, truncated)

        print("Time add: ", (time() - t_begin) / self.n_runs)

    def time_sample_batch(self) -> None:
        replay_buffer = ReplayBuffer(self.max_size, self.batch_size, self.state_shape, self.state_dtype, self.identity)
        batch_key = self.key

        for _ in range(10):
            batch_key, key = jax.random.split(batch_key)
            state = np.array(jax.random.randint(key, replay_buffer.state_shape, 1, 256, replay_buffer.state_dtype))
            action = np.array(jax.random.randint(key, (), 0, 10, replay_buffer.action_dtype))
            reward = np.array(jax.random.uniform(key, (), replay_buffer.reward_dtype, 0, 1000))
            absorbing = np.array(jax.random.randint(key, (), 0, 2), dtype=replay_buffer.absorbing_dtype)
            batch_key, key_truncateds, keys_next_states = jax.random.split(self.key, 3)
            truncated = np.array(jax.random.randint(key_truncateds, (), 0, 2), dtype=replay_buffer.absorbing_dtype)
            next_state = np.array(
                jax.random.randint(keys_next_states, replay_buffer.state_shape, 1, 256, replay_buffer.state_dtype)
            )

            replay_buffer.add(state, action, reward, next_state, absorbing, truncated)

        t_begin = time()

        for _ in range(self.n_runs):
            batch_key, key = jax.random.split(batch_key)
            replay_buffer.sample_random_batch(batch_key)

        print("Time sample batch: ", (time() - t_begin) / self.n_runs)


class TimeNStepReplayBuffer:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = RANDOM_SEED
        print(f"random seed {self.random_seed}")
        self.n_runs = 3000
        self.key = jax.random.PRNGKey(self.random_seed)
        self.n_steps_return = jax.random.randint(self.key, (), minval=1, maxval=10)
        print(f"{self.n_steps_return}-step return")
        self.gamma = np.float32(jax.random.uniform(self.key))
        self.max_size = jax.random.randint(self.key, (), minval=1, maxval=1000)
        self.batch_size = jax.random.randint(self.key, (), minval=1, maxval=self.max_size)
        self.state_shape = (4, 84, 84)
        self.state_dtype = np.uint8
        self.path = "tests/replay_buffer"
        self.identity = lambda x: x

    def time_add(self) -> None:
        replay_buffer = NStepReplayBuffer(
            self.n_steps_return,
            self.gamma,
            self.max_size,
            self.batch_size,
            self.state_shape,
            self.state_dtype,
            self.identity,
        )
        batch_key = self.key

        t_begin = time()

        for _ in range(self.n_runs):
            batch_key, key = jax.random.split(batch_key)
            state = np.array(jax.random.randint(key, replay_buffer.state_shape, 0, 256, replay_buffer.state_dtype))
            action = np.array(jax.random.randint(key, (), 0, 10, replay_buffer.action_dtype))
            reward = np.array(jax.random.uniform(key, (), replay_buffer.reward_dtype, 0, 1000))
            absorbing = np.array(jax.random.randint(key, (), 0, 2), dtype=replay_buffer.absorbing_dtype)
            batch_key, key_truncateds, keys_next_states = jax.random.split(self.key, 3)
            truncated = np.array(jax.random.randint(key_truncateds, (), 0, 2), dtype=replay_buffer.absorbing_dtype)
            next_state = np.array(
                jax.random.randint(keys_next_states, replay_buffer.state_shape, 1, 256, replay_buffer.state_dtype)
            )

            replay_buffer.add(state, action, reward, next_state, absorbing, truncated)

        print("Time add: ", (time() - t_begin) / self.n_runs)
