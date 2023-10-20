import unittest
import jax
import jax.numpy as jnp
import numpy as np

from idqn.sample_collection import IDX_RB
from idqn.networks.q_architectures import AtariREM


class TestAtariREM(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        print(f"random seed {self.random_seed}")
        self.key = jax.random.PRNGKey(self.random_seed)
        self.state_shape = (84, 84, 4)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.gamma = jax.random.uniform(self.key)

    def test_output(self) -> None:
        q = AtariREM(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)

        state = jax.random.uniform(self.key, self.state_shape, minval=-1, maxval=1)
        state_copy = state.copy()

        output = q.apply(q.params, state)
        output_batch = q.apply(q.params, jax.random.uniform(self.key, (50,) + self.state_shape, minval=-1, maxval=1))

        self.assertGreater(np.linalg.norm(output), 0)
        self.assertGreater(np.linalg.norm(output_batch), 0)

        self.assertEqual(output.shape, (1, self.n_actions))
        self.assertEqual(output_batch.shape, (50, self.n_actions))

        # test if the input has been changed
        self.assertEqual(np.linalg.norm(state - state_copy), 0)
        self.assertEqual(state.shape, state_copy.shape)

    def test_compute_target(self) -> None:
        q = AtariREM(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)

        rewards = jax.random.uniform(self.key, (10,), minval=-1, maxval=1)
        terminals = jax.random.randint(self.key, (10,), 0, 2)
        next_states = jax.random.uniform(self.key, (10,) + self.state_shape, minval=-1, maxval=1)
        samples = (
            0,  # state
            0,  # action
            jnp.array(rewards, dtype=jnp.float32),  # reward
            jnp.array(next_states, dtype=jnp.float32),  # next_state
            0,  # next_action
            0,  # next_reward
            jnp.array(terminals, dtype=jnp.bool_),  # terminal
        )

        computed_targets = q.compute_target(q.params, samples)

        for idx_sample in range(10):
            target = rewards[idx_sample] + (1 - terminals[idx_sample]) * self.gamma * jnp.max(
                q.apply(q.params, next_states[idx_sample])
            )
            self.assertAlmostEqual(computed_targets[idx_sample], target, places=6)

    def test_loss(self) -> None:
        q = AtariREM(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)

        states = jax.random.uniform(self.key, (10,) + self.state_shape, minval=-1, maxval=1)
        actions = jax.random.randint(self.key, (10,), minval=0, maxval=self.n_actions)
        key, _ = jax.random.split(self.key)
        rewards = jax.random.uniform(key, (10,), minval=-1, maxval=1)
        terminals = jax.random.randint(key, (10,), 0, 2)
        next_states = jax.random.uniform(key, (10,) + self.state_shape, minval=-1, maxval=1)
        samples = (
            jnp.array(states, dtype=jnp.float32),  # state
            jnp.array(actions, dtype=jnp.int8),  # action
            jnp.array(rewards, dtype=jnp.float32),  # reward
            jnp.array(next_states, dtype=jnp.float32),  # next_state
            0,  # next_action
            0,  # next_reward
            jnp.array(terminals, dtype=jnp.bool_),  # terminal
        )

        computed_loss = q.loss(q.params, q.params, samples)

        targets = q.compute_target(q.params, samples)
        predictions = np.zeros(10) * np.nan

        for idx_sample in range(10):
            predictions[idx_sample] = q.apply(q.params, states[idx_sample])[0, actions.astype(jnp.int8)[idx_sample]]

        self.assertAlmostEqual(computed_loss, np.square(targets - predictions).mean(), places=6)

    def test_random_action(self) -> None:
        q = AtariREM(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)

        self.assertEqual(q.random_action(self.key).dtype, jnp.int8)

    def test_best_action(self) -> None:
        q = AtariREM(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)

        state = jax.random.uniform(self.key, self.state_shape, minval=-1, maxval=1)

        computed_best_action = q.best_action(q.params, state)

        best_action = jnp.argmax(q.apply(q.params, state)[0]).astype(jnp.int8)
        self.assertEqual(best_action, computed_best_action)
