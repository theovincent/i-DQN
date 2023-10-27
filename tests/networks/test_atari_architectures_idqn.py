import unittest
import jax
import jax.numpy as jnp
import numpy as np

from idqn.networks.architectures.idqn import AtariiDQN


class TestAtariiDQN(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        print(f"random seed {self.random_seed}")
        self.key = jax.random.PRNGKey(self.random_seed)
        self.n_heads = int(jax.random.randint(self.key, (), minval=2, maxval=50))
        self.state_shape = (84, 84, 4)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.cumulative_gamma = jax.random.uniform(self.key)
        self.head_behaviorial_probability = jax.random.uniform(self.key, (self.n_heads,), minval=1, maxval=10)

    def test_output(self) -> None:
        q = AtariiDQN(
            self.n_heads,
            self.state_shape,
            self.n_actions,
            self.cumulative_gamma,
            self.key,
            self.head_behaviorial_probability,
            None,
            None,
            None,
            None,
            None,
            shared_network=True,
        )

        state = jax.random.uniform(self.key, self.state_shape, minval=-1, maxval=1)
        state_copy = state.copy()

        output = q.apply(q.params, state)
        output_batch = q.apply(q.params, jax.random.uniform(self.key, (50,) + self.state_shape, minval=-1, maxval=1))

        self.assertGreater(np.linalg.norm(output), 0)
        self.assertGreater(np.linalg.norm(output_batch), 0)

        self.assertEqual(output.shape, (1, self.n_heads, self.n_actions))
        self.assertEqual(output_batch.shape, (50, self.n_heads, self.n_actions))

        # test if the input has been changed
        self.assertEqual(np.linalg.norm(state - state_copy), 0)
        self.assertEqual(state.shape, state_copy.shape)

    def test_compute_target(self) -> None:
        q = AtariiDQN(
            self.n_heads,
            self.state_shape,
            self.n_actions,
            self.cumulative_gamma,
            self.key,
            self.head_behaviorial_probability,
            None,
            None,
            None,
            None,
            None,
            shared_network=True,
        )

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
            0,  # indices
        )

        computed_targets = q.compute_target(q.params, samples)

        for idx_sample in range(10):
            for idx_head in range(self.n_heads):
                target = rewards[idx_sample] + (1 - terminals[idx_sample]) * self.cumulative_gamma * jnp.max(
                    q.apply(q.params, next_states[idx_sample])[:, idx_head]
                )
                self.assertAlmostEqual(computed_targets[idx_sample, idx_head], target, places=6)

    def test_loss(self) -> None:
        q = AtariiDQN(
            self.n_heads,
            self.state_shape,
            self.n_actions,
            self.cumulative_gamma,
            self.key,
            self.head_behaviorial_probability,
            None,
            None,
            None,
            None,
            None,
            shared_network=True,
        )
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
            jax.random.split(self.key)[0],  # key
            jax.random.split(self.key)[1],  # next_key
        )

        computed_loss = q.loss(q.params, q.params, samples)

        targets = q.compute_target(q.params, samples)
        predictions = np.zeros((10, self.n_heads)) * np.nan

        for idx_sample in range(10):
            for idx_head in range(self.n_heads):
                predictions[idx_sample, idx_head] = q.apply(q.params, states[idx_sample])[
                    0, idx_head, actions.astype(jnp.int8)[idx_sample]
                ]

        self.assertAlmostEqual(computed_loss, np.square(targets[:, :-1] - predictions[:, 1:]).mean(), places=6)

    def test_best_action(self) -> None:
        q = AtariiDQN(
            self.n_heads,
            self.state_shape,
            self.n_actions,
            self.cumulative_gamma,
            self.key,
            jnp.zeros(self.n_heads).at[-1].set(1),
            None,
            None,
            None,
            None,
            None,
            shared_network=True,
        )
        state = jax.random.uniform(self.key, self.state_shape, minval=-1, maxval=1)

        computed_best_action = q.best_action(q.params, state, key=self.key)

        # -1 since head behavioral policy equals to [0, ..., 0, 1]
        best_action = jnp.argmax(q.apply(q.params, state)[0, -1]).astype(jnp.int8)
        self.assertEqual(best_action, computed_best_action)

    def test_rolling_step(self) -> None:
        q = AtariiDQN(
            self.n_heads,
            self.state_shape,
            self.n_actions,
            self.cumulative_gamma,
            self.key,
            self.head_behaviorial_probability,
            None,
            None,
            None,
            None,
            None,
            shared_network=True,
        )
        state = jax.random.uniform(self.key, (50,) + self.state_shape, minval=-1, maxval=1)

        output = q.apply(q.params, state)

        q.params = q.rolling_step(q.params)

        forward_output = q.apply(q.params, state)

        self.assertAlmostEqual(np.linalg.norm(forward_output[:, :-1] - output[:, 1:]), 0, places=5)
