import unittest
import jax
import jax.numpy as jnp
import numpy as np

from idqn.sample_collection import IDX_RB
from idqn.networks.architectures.iqn import AtariIQN


class TestAtariIQN(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        print(f"random seed {self.random_seed}")
        self.key = jax.random.PRNGKey(self.random_seed)
        self.state_shape = (84, 84, 4)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.gamma = jax.random.uniform(self.key)

    def test_output(self) -> None:
        q = AtariIQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)

        state = jax.random.uniform(self.key, self.state_shape, minval=-1, maxval=1)
        state_copy = state.copy()

        output, quantiles = q.apply_n_quantiles(q.params, state, self.key)
        output_batch, batch_quantiles = q.apply_n_quantiles_target(
            q.params, jax.random.uniform(self.key, (50,) + self.state_shape, minval=-1, maxval=1), self.key
        )

        self.assertGreater(np.linalg.norm(output), 0)
        self.assertGreater(np.linalg.norm(output_batch), 0)

        self.assertEqual(quantiles.shape, (1, q.n_quantiles))
        self.assertEqual(batch_quantiles.shape, (50, q.n_quantiles_policy + q.n_quantiles_target))

        self.assertEqual(output.shape, (1, q.n_quantiles, self.n_actions))
        self.assertEqual(output_batch.shape, (50, q.n_quantiles_policy + q.n_quantiles_target, self.n_actions))

        # test if the input has been changed
        self.assertEqual(np.linalg.norm(state - state_copy), 0)
        self.assertEqual(state.shape, state_copy.shape)

    def test_compute_target(self) -> None:
        q = AtariIQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)

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
            jax.random.split(self.key)[0],  # key
            jax.random.split(self.key)[1],  # next_key
        )
        computed_targets = q.compute_target(q.params, samples)

        quantiles_policy_targets, _ = q.network.apply(
            q.target_params, next_states, samples[IDX_RB["next_key"]], q.n_quantiles_policy + q.n_quantiles_target
        )
        quantiles_policy, quantiles_targets = (
            quantiles_policy_targets[:, : q.n_quantiles_policy],
            quantiles_policy_targets[:, q.n_quantiles_policy :],
        )

        for idx_sample in range(10):
            value_policy = jnp.mean(quantiles_policy[idx_sample], axis=0)
            action = jnp.argmax(value_policy)

            target = (
                rewards[idx_sample]
                + (1 - terminals[idx_sample]) * self.gamma * quantiles_targets[idx_sample, :, action]
            )
            self.assertAlmostEqual(jnp.linalg.norm(computed_targets[idx_sample] - target), 0, places=6)

    def test_loss(self) -> None:
        q = AtariIQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)
        q.n_quantiles = 13
        q.n_quantiles_target = 9

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
            0,  # indices
            jax.random.split(self.key)[0],  # key
            jax.random.split(self.key)[1],  # next_key
        )
        computed_loss = q.loss(q.params, q.params, samples)

        targets = q.compute_target(q.params, samples)
        predictions, quantiles = q.network.apply(q.params, states, samples[IDX_RB["key"]], q.n_quantiles)

        loss = 0

        for idx_sample in range(10):
            for idx_quantile in range(q.n_quantiles):
                for idx_quantile_target in range(q.n_quantiles_target):
                    bellman_error = (
                        targets[idx_sample, idx_quantile_target]
                        - predictions[idx_sample, idx_quantile, actions[idx_sample]]
                    )
                    huber_loss = (
                        1 / 2 * bellman_error**2 if jnp.abs(bellman_error) < 1 else jnp.abs(bellman_error) - 1 / 2
                    )
                    loss += jnp.abs(quantiles[idx_sample, idx_quantile] - (bellman_error < 0)) * huber_loss

        loss /= 10 * q.n_quantiles_target

        self.assertAlmostEqual(computed_loss, loss, places=5)

    def test_best_action(self) -> None:
        q = AtariIQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)

        state = jax.random.uniform(self.key, self.state_shape, minval=-1, maxval=1)

        computed_best_action = q.best_action(q.params, state, key=self.key)

        quantiles_policy, _ = q.network.apply(q.params, state, self.key, q.n_quantiles_policy)
        value_policy = jnp.mean(quantiles_policy, axis=1)[0]
        best_action = jnp.argmax(value_policy)

        self.assertEqual(best_action, computed_best_action)
