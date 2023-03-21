import os
import unittest
import jax
import numpy as np

from idqn.sample_collection.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        print(f"random seed {self.random_seed}")
        self.key = jax.random.PRNGKey(self.random_seed)
        self.max_size = jax.random.randint(self.key, (), minval=1, maxval=1000)
        self.batch_size = jax.random.randint(self.key, (), minval=1, maxval=self.max_size)
        self.state_shape = (4, 84, 84)
        self.state_dtype = np.uint8
        self.path = "tests/replay_buffer"
        self.identity = lambda x: x

    def test_add(self) -> None:
        replay_buffer = ReplayBuffer(self.max_size, self.batch_size, self.state_shape, self.state_dtype, self.identity)

        state = np.array(jax.random.randint(self.key, replay_buffer.state_shape, 0, 256, replay_buffer.state_dtype))
        action = np.array(jax.random.randint(self.key, (), 0, 10, replay_buffer.action_dtype))
        reward = np.array(jax.random.uniform(self.key, (), replay_buffer.reward_dtype, 0, 1000))
        absorbing = np.array(jax.random.randint(self.key, (), 0, 2), dtype=replay_buffer.absorbing_dtype)
        key, _ = jax.random.split(self.key)
        next_state = np.array(jax.random.randint(key, replay_buffer.state_shape, 0, 256, replay_buffer.state_dtype))

        replay_buffer.add(state, action, reward, next_state, absorbing)

        # Check if only the reference have been stored
        state_copy = state.copy()
        next_state_copy = next_state.copy()
        state += 10
        next_state += 10

        self.assertEqual(np.linalg.norm(replay_buffer.states[0] - state_copy), 0)
        self.assertEqual(replay_buffer.actions[0], action)
        self.assertEqual(replay_buffer.rewards[0], reward)
        self.assertEqual(np.linalg.norm(replay_buffer.next_states[0] - next_state_copy), 0)
        self.assertEqual(replay_buffer.absorbings[0], absorbing)

        self.assertEqual(replay_buffer.states.shape, (self.max_size,) + self.state_shape)
        self.assertEqual(replay_buffer.actions.shape, (self.max_size,))
        self.assertEqual(replay_buffer.rewards.shape, (self.max_size,))
        self.assertEqual(replay_buffer.next_states.shape, (self.max_size,) + self.state_shape)

        self.assertEqual(replay_buffer.states[0].dtype, replay_buffer.state_dtype)
        self.assertEqual(replay_buffer.actions[0].dtype, replay_buffer.action_dtype)
        self.assertEqual(replay_buffer.rewards[0].dtype, replay_buffer.reward_dtype)
        self.assertEqual(replay_buffer.next_states[0].dtype, replay_buffer.state_dtype)

    def test_sample_batch(self) -> None:
        replay_buffer = ReplayBuffer(self.max_size, self.batch_size, self.state_shape, self.state_dtype, self.identity)
        key = self.key

        for _ in range(10):
            key, _ = jax.random.split(key)
            state = np.array(jax.random.randint(key, replay_buffer.state_shape, 1, 256, replay_buffer.state_dtype))
            action = np.array(jax.random.randint(key, (), 0, 10, replay_buffer.action_dtype))
            reward = np.array(jax.random.uniform(self.key, (), replay_buffer.reward_dtype, 0, 1000))
            absorbing = np.array(jax.random.randint(key, (), 0, 2), dtype=replay_buffer.absorbing_dtype)
            key, _ = jax.random.split(key)
            next_state = np.array(jax.random.randint(key, replay_buffer.state_shape, 1, 256, replay_buffer.state_dtype))

            replay_buffer.add(state, action, reward, next_state, absorbing)

        batch = replay_buffer.sample_random_batch(self.key)

        for idx_in_batch in range(self.batch_size):
            self.assertIn(batch["state"][idx_in_batch], replay_buffer.states)
            self.assertIn(batch["action"][idx_in_batch], replay_buffer.actions)
            self.assertIn(batch["reward"][idx_in_batch], replay_buffer.rewards)
            self.assertIn(batch["next_state"][idx_in_batch], replay_buffer.next_states)
            self.assertIn(batch["absorbing"][idx_in_batch], replay_buffer.absorbings)

            self.assertNotEqual(np.linalg.norm(batch["state"][idx_in_batch]), 0)

    def test_save_load(self) -> None:
        replay_buffer = ReplayBuffer(self.max_size, self.batch_size, self.state_shape, self.state_dtype, self.identity)
        key = self.key

        for _ in range(10):
            key, _ = jax.random.split(key)
            state = np.array(jax.random.randint(key, replay_buffer.state_shape, 1, 256, replay_buffer.state_dtype))
            action = np.array(jax.random.randint(key, (), 0, 10, replay_buffer.action_dtype))
            reward = np.array(jax.random.uniform(self.key, (), replay_buffer.reward_dtype, 0, 1000))
            absorbing = np.array(jax.random.randint(key, (), 0, 2), dtype=replay_buffer.absorbing_dtype)
            key, _ = jax.random.split(key)
            next_state = np.array(jax.random.randint(key, replay_buffer.state_shape, 1, 256, replay_buffer.state_dtype))

            replay_buffer.add(state, action, reward, next_state, absorbing)

        replay_buffer.save(self.path)

        replay_buffer_bis = ReplayBuffer(
            self.max_size, self.batch_size, self.state_shape, self.state_dtype, self.identity
        )
        replay_buffer_bis.load(self.path)

        self.assertEqual(np.linalg.norm(replay_buffer.states - replay_buffer_bis.states), 0)
        self.assertEqual(np.linalg.norm(replay_buffer.actions - replay_buffer_bis.actions), 0)
        self.assertEqual(np.linalg.norm(replay_buffer.rewards - replay_buffer_bis.rewards), 0)
        self.assertEqual(
            np.linalg.norm(replay_buffer.next_states - replay_buffer_bis.next_states),
            0,
            f"random seed {self.random_seed}",
        )
        self.assertEqual(np.sum(~(replay_buffer.absorbings == replay_buffer_bis.absorbings)), 0)
        self.assertEqual(replay_buffer.idx, replay_buffer_bis.idx)
        self.assertEqual(replay_buffer.len, replay_buffer_bis.len)

        self.assertEqual(replay_buffer.states.shape, replay_buffer_bis.states.shape)
        self.assertEqual(replay_buffer.actions.shape, replay_buffer_bis.actions.shape)
        self.assertEqual(replay_buffer.rewards.shape, replay_buffer_bis.rewards.shape)
        self.assertEqual(replay_buffer.next_states.shape, replay_buffer_bis.next_states.shape)
        self.assertEqual(
            replay_buffer.absorbing_dtype.shape,
            replay_buffer_bis.absorbing_dtype.shape,
            f"random seed {self.random_seed}",
        )

        self.assertEqual(replay_buffer.states.dtype, replay_buffer_bis.states.dtype)
        self.assertEqual(replay_buffer.actions.dtype, replay_buffer_bis.actions.dtype)
        self.assertEqual(replay_buffer.rewards.dtype, replay_buffer_bis.rewards.dtype)
        self.assertEqual(replay_buffer.next_states.dtype, replay_buffer_bis.next_states.dtype)
        self.assertEqual(replay_buffer.absorbings.dtype, replay_buffer_bis.absorbings.dtype)

        os.remove(self.path + "_states.npy")
        os.remove(self.path + "_actions.npy")
        os.remove(self.path + "_rewards.npy")
        os.remove(self.path + "_next_states.npy")
        os.remove(self.path + "_absorbings.npy")
        os.remove(self.path + "_idx.npy")
        os.remove(self.path + "_len.npy")
