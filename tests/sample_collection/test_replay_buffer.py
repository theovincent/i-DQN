import os
import unittest
import jax
import numpy as np

from idqn.sample_collection.replay_buffer import ReplayBuffer, NStepReplayBuffer


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

        replay_buffer.add(state, action, reward, next_state, absorbing, None)  # Truncated is not relevant here

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
            reward = np.array(jax.random.uniform(key, (), replay_buffer.reward_dtype, 0, 1000))
            absorbing = np.array(jax.random.randint(key, (), 0, 2), dtype=replay_buffer.absorbing_dtype)
            key, _ = jax.random.split(key)
            next_state = np.array(jax.random.randint(key, replay_buffer.state_shape, 1, 256, replay_buffer.state_dtype))

            replay_buffer.add(state, action, reward, next_state, absorbing, None)  # Truncated is not relevant here

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

            replay_buffer.add(state, action, reward, next_state, absorbing, None)  # Truncated is not relevant here

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


class TestNStepReplayBuffer(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        print(f"random seed {self.random_seed}")
        self.key = jax.random.PRNGKey(self.random_seed)
        self.n_steps_return = jax.random.randint(self.key, (), minval=1, maxval=10)
        self.gamma = np.float32(jax.random.uniform(self.key))
        self.max_size = jax.random.randint(self.key, (), minval=50, maxval=1000)
        self.batch_size = jax.random.randint(self.key, (), minval=1, maxval=self.max_size)
        self.state_shape = (4, 84, 84)
        self.state_dtype = np.uint8
        self.identity = lambda x: x

    def test_add(self) -> None:
        replay_buffer = NStepReplayBuffer(
            self.n_steps_return,
            self.gamma,
            self.max_size,
            self.batch_size,
            self.state_shape,
            self.state_dtype,
            self.identity,
        )
        n_samples = 50

        states = np.array(
            jax.random.randint(self.key, (n_samples,) + replay_buffer.state_shape, 0, 256, replay_buffer.state_dtype)
        )
        actions = np.array(jax.random.randint(self.key, (n_samples,), 0, 10, replay_buffer.action_dtype))
        rewards = np.array(jax.random.uniform(self.key, (n_samples,), replay_buffer.reward_dtype, 0, 1000))
        absorbings = np.array(jax.random.randint(self.key, (n_samples,), 0, 2), dtype=replay_buffer.absorbing_dtype)
        # We enforce the last sample to be absorbing so that n_samples are added to the replay buffer,
        # i.e. at the end of an episode, the replay buffer should add all samples from the step buffer to the "real" buffer.
        absorbings[-1] = True
        key_truncateds, keys_next_states = jax.random.split(self.key)
        truncateds = np.array(
            jax.random.randint(key_truncateds, (n_samples,), 0, 2), dtype=replay_buffer.absorbing_dtype
        )
        next_states = np.array(
            jax.random.randint(
                keys_next_states, (n_samples,) + replay_buffer.state_shape, 0, 256, replay_buffer.state_dtype
            )
        )

        for idx_sample in range(n_samples):
            replay_buffer.add(
                states[idx_sample],
                actions[idx_sample],
                rewards[idx_sample],
                next_states[idx_sample],
                absorbings[idx_sample],
                truncateds[idx_sample],
            )

        for idx_sample in range(n_samples):
            self.assertEqual(np.linalg.norm(states[idx_sample] - replay_buffer.states[idx_sample]), 0)
            self.assertEqual(np.linalg.norm(actions[idx_sample] - replay_buffer.actions[idx_sample]), 0)

            # Compute the number of n-steps that the sample should have access to
            reward = rewards[idx_sample]
            i = 1
            while i < self.n_steps_return and not absorbings[idx_sample + i - 1] and not truncateds[idx_sample + i - 1]:
                reward += self.gamma**i * rewards[idx_sample + i]
                i += 1

            self.assertAlmostEqual(reward, replay_buffer.rewards[idx_sample], places=3)
            self.assertEqual(np.linalg.norm(next_states[idx_sample + i - 1] - replay_buffer.next_states[idx_sample]), 0)
            self.assertEqual(absorbings[idx_sample + i - 1], replay_buffer.absorbings[idx_sample])
