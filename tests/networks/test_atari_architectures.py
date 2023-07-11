import unittest
import jax
import jax.numpy as jnp
import numpy as np

from idqn.networks.q_architectures import AtariDQN, AtariIQN, AtariiDQN


class TestAtariDQN(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        print(f"random seed {self.random_seed}")
        self.key = jax.random.PRNGKey(self.random_seed)
        self.state_shape = (4, 84, 84)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.gamma = jax.random.uniform(self.key)

    def test_output(self) -> None:
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)

        state = jax.random.uniform(self.key, self.state_shape)
        state_copy = state.copy()

        output = q(q.params, state)
        output_batch = q(q.params, jax.random.uniform(self.key, (50,) + self.state_shape))

        self.assertGreater(np.linalg.norm(output), 0)
        self.assertGreater(np.linalg.norm(output_batch), 0)

        self.assertEqual(output.shape, (1, self.n_actions))
        self.assertEqual(output_batch.shape, (50, self.n_actions))

        # test if the input has been changed
        self.assertEqual(np.linalg.norm(state - state_copy), 0)
        self.assertEqual(state.shape, state_copy.shape)

    def test_compute_target(self) -> None:
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)

        rewards = jax.random.uniform(self.key, (10,))
        absorbings = jax.random.randint(self.key, (10,), 0, 2)
        next_states = jax.random.uniform(self.key, (10,) + self.state_shape)
        samples = {
            "reward": jnp.array(rewards, dtype=jnp.float32),
            "next_state": jnp.array(next_states, dtype=jnp.float32),
            "absorbing": jnp.array(absorbings, dtype=jnp.bool_),
        }

        computed_targets = q.compute_target(q.params, samples)

        for idx_sample in range(10):
            target = rewards[idx_sample] + (1 - absorbings[idx_sample]) * self.gamma * jnp.max(
                q(q.params, next_states[idx_sample])
            )
            self.assertAlmostEqual(computed_targets[idx_sample], target)

    def test_loss(self) -> None:
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)

        states = jax.random.uniform(self.key, (10,) + self.state_shape)
        actions = jax.random.randint(self.key, (10,), minval=0, maxval=self.n_actions)
        key, _ = jax.random.split(self.key)
        rewards = jax.random.uniform(key, (10,))
        absorbings = jax.random.randint(key, (10,), 0, 2)
        next_states = jax.random.uniform(key, (10,) + self.state_shape)
        samples = {
            "state": jnp.array(states, dtype=jnp.float32),
            "action": jnp.array(actions, dtype=jnp.int8),
            "reward": jnp.array(rewards, dtype=jnp.float32),
            "next_state": jnp.array(next_states, dtype=jnp.float32),
            "absorbing": jnp.array(absorbings, dtype=jnp.bool_),
        }

        computed_loss = q.loss(q.params, q.params, samples)

        targets = q.compute_target(q.params, samples)
        predictions = np.zeros(10) * np.nan

        for idx_sample in range(10):
            predictions[idx_sample] = q(q.params, states[idx_sample])[0, actions.astype(jnp.int8)[idx_sample]]

        self.assertAlmostEqual(computed_loss, np.square(targets - predictions).mean(), places=6)

    def test_random_action(self) -> None:
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)

        self.assertEqual(q.random_action(self.key).dtype, jnp.int8)

    def test_best_action(self) -> None:
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)

        state = jax.random.uniform(self.key, self.state_shape)

        computed_best_action = q.best_action(q.params, state, None)

        best_action = jnp.argmax(q(q.params, state)[0]).astype(jnp.int8)
        self.assertEqual(best_action, computed_best_action)


class TestAtariIQN(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        print(f"random seed {self.random_seed}")
        self.key = jax.random.PRNGKey(self.random_seed)
        self.state_shape = (4, 84, 84)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.gamma = jax.random.uniform(self.key)

    def test_output(self) -> None:
        q = AtariIQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)

        state = jax.random.uniform(self.key, self.state_shape)
        state_copy = state.copy()

        output, quantiles = q.apply_n_quantiles(q.params, state, self.key)
        output_batch, batch_quantiles = q.apply_n_quantiles_target(
            q.params, jax.random.uniform(self.key, (50,) + self.state_shape), self.key
        )

        self.assertGreater(np.linalg.norm(output), 0)
        self.assertGreater(np.linalg.norm(output_batch), 0)

        self.assertEqual(quantiles.shape, (1, q.n_quantiles))
        self.assertEqual(batch_quantiles.shape, (50, q.n_quantiles_target))

        self.assertEqual(output.shape, (1, q.n_quantiles, self.n_actions))
        self.assertEqual(output_batch.shape, (50, q.n_quantiles_target, self.n_actions))

        # test if the input has been changed
        self.assertEqual(np.linalg.norm(state - state_copy), 0)
        self.assertEqual(state.shape, state_copy.shape)

    def test_compute_target(self) -> None:
        q = AtariIQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)

        rewards = jax.random.uniform(self.key, (10,))
        absorbings = jax.random.randint(self.key, (10,), 0, 2)
        next_states = jax.random.uniform(self.key, (10,) + self.state_shape)
        samples = {
            "reward": jnp.array(rewards, dtype=jnp.float32),
            "next_state": jnp.array(next_states, dtype=jnp.float32),
            "absorbing": jnp.array(absorbings, dtype=jnp.bool_),
        }
        samples["key"], samples["next_key"], samples["policy_key"] = jax.random.split(self.key, 3)

        computed_targets = q.compute_target(q.params, samples)

        quantiles_policy, _ = q.apply_n_quantiles_policy(q.target_params, next_states, samples["policy_key"])
        quantiles_targets, _ = q.apply_n_quantiles_target(q.target_params, next_states, samples["next_key"])

        for idx_sample in range(10):
            value_policy = jnp.mean(quantiles_policy[idx_sample], axis=0)
            action = jnp.argmax(value_policy)

            target = (
                rewards[idx_sample]
                + (1 - absorbings[idx_sample]) * self.gamma * quantiles_targets[idx_sample, :, action]
            )
            self.assertAlmostEqual(jnp.linalg.norm(computed_targets[idx_sample] - target), 0)

    def test_loss(self) -> None:
        q = AtariIQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)
        q.n_quantiles = 13
        q.n_quantiles_target = 9

        states = jax.random.uniform(self.key, (10,) + self.state_shape)
        actions = jax.random.randint(self.key, (10,), minval=0, maxval=self.n_actions)
        key, _ = jax.random.split(self.key)
        rewards = jax.random.uniform(key, (10,))
        absorbings = jax.random.randint(key, (10,), 0, 2)
        next_states = jax.random.uniform(key, (10,) + self.state_shape)
        samples = {
            "state": jnp.array(states, dtype=jnp.float32),
            "action": jnp.array(actions, dtype=jnp.int8),
            "reward": jnp.array(rewards, dtype=jnp.float32),
            "next_state": jnp.array(next_states, dtype=jnp.float32),
            "absorbing": jnp.array(absorbings, dtype=jnp.bool_),
        }
        samples["key"], samples["next_key"], samples["policy_key"] = jax.random.split(self.key, 3)

        computed_loss = q.loss(q.params, q.params, samples)

        targets = q.compute_target(q.params, samples)
        predictions, quantiles = q.apply_n_quantiles(q.params, states, samples["key"])

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
                    loss += (quantiles[idx_sample, idx_quantile] - (bellman_error < 0)) * huber_loss
        loss /= 10 * q.n_quantiles_target

        self.assertAlmostEqual(computed_loss, loss, places=5)

    def test_best_action(self) -> None:
        q = AtariIQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)

        state = jax.random.uniform(self.key, self.state_shape)

        computed_best_action = q.best_action(q.params, state, self.key)

        quantiles_policy, _ = q.apply_n_quantiles_policy(q.params, state, self.key)
        value_policy = jnp.mean(quantiles_policy, axis=1)[0]
        best_action = jnp.argmax(value_policy)

        self.assertEqual(best_action, computed_best_action)


class TestAtariiDQN(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        print(f"random seed {self.random_seed}")
        self.key = jax.random.PRNGKey(self.random_seed)
        self.n_heads = jax.random.randint(self.key, (), minval=2, maxval=50)
        self.importance_iteration = jax.random.uniform(self.key, (self.n_heads - 1,), minval=1, maxval=10)
        self.state_shape = (4, 84, 84)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.gamma = jax.random.uniform(self.key)
        self.head_behaviorial_probability = jax.random.uniform(self.key, (self.n_heads,), minval=1, maxval=10)

    def test_output(self) -> None:
        q = AtariiDQN(
            self.importance_iteration,
            self.state_shape,
            self.n_actions,
            self.gamma,
            self.key,
            self.head_behaviorial_probability,
            None,
            None,
            None,
            None,
            None,
        )

        state = jax.random.uniform(self.key, self.state_shape)
        state_copy = state.copy()

        output = q(q.params, state)
        output_batch = q(q.params, jax.random.uniform(self.key, (50,) + self.state_shape))

        self.assertGreater(np.linalg.norm(output), 0)
        self.assertGreater(np.linalg.norm(output_batch), 0)

        self.assertEqual(output.shape, (1, self.n_heads, self.n_actions))
        self.assertEqual(output_batch.shape, (50, self.n_heads, self.n_actions))

        # test if the input has been changed
        self.assertEqual(np.linalg.norm(state - state_copy), 0)
        self.assertEqual(state.shape, state_copy.shape)

    def test_compute_target(self) -> None:
        q = AtariiDQN(
            self.importance_iteration,
            self.state_shape,
            self.n_actions,
            self.gamma,
            self.key,
            self.head_behaviorial_probability,
            None,
            None,
            None,
            None,
            None,
        )

        rewards = jax.random.uniform(self.key, (10,))
        absorbings = jax.random.randint(self.key, (10,), 0, 2)
        next_states = jax.random.uniform(self.key, (10,) + self.state_shape)
        samples = {
            "reward": jnp.array(rewards, dtype=jnp.float32),
            "next_state": jnp.array(next_states, dtype=jnp.float32),
            "absorbing": jnp.array(absorbings, dtype=jnp.bool_),
        }

        computed_targets = q.compute_target(q.params, samples)

        for idx_sample in range(10):
            for idx_head in range(self.n_heads):
                target = rewards[idx_sample] + (1 - absorbings[idx_sample]) * self.gamma * jnp.max(
                    q(q.params, next_states[idx_sample])[:, idx_head]
                )
                self.assertAlmostEqual(computed_targets[idx_sample, idx_head], target)

    def test_loss(self) -> None:
        q = AtariiDQN(
            jnp.ones(self.n_heads - 1),
            self.state_shape,
            self.n_actions,
            self.gamma,
            self.key,
            self.head_behaviorial_probability,
            None,
            None,
            None,
            None,
            None,
        )
        states = jax.random.uniform(self.key, (10,) + self.state_shape)
        actions = jax.random.randint(self.key, (10,), minval=0, maxval=self.n_actions)
        key, _ = jax.random.split(self.key)
        rewards = jax.random.uniform(key, (10,))
        absorbings = jax.random.randint(key, (10,), 0, 2)
        next_states = jax.random.uniform(key, (10,) + self.state_shape)
        samples = {
            "state": jnp.array(states, dtype=jnp.float32),
            "action": jnp.array(actions, dtype=jnp.int8),
            "reward": jnp.array(rewards, dtype=jnp.float32),
            "next_state": jnp.array(next_states, dtype=jnp.float32),
            "absorbing": jnp.array(absorbings, dtype=jnp.bool_),
        }

        computed_loss = q.loss(q.params, q.params, samples)

        targets = q.compute_target(q.params, samples)
        predictions = np.zeros((10, self.n_heads)) * np.nan

        for idx_sample in range(10):
            for idx_head in range(self.n_heads):
                predictions[idx_sample, idx_head] = q(q.params, states[idx_sample])[
                    0, idx_head, actions.astype(jnp.int8)[idx_sample]
                ]

        self.assertAlmostEqual(computed_loss, np.square(targets[:, :-1] - predictions[:, 1:]).mean(), places=6)

    def test_best_action(self) -> None:
        q = AtariiDQN(
            self.importance_iteration,
            self.state_shape,
            self.n_actions,
            self.gamma,
            self.key,
            jnp.zeros(self.n_heads).at[-1].set(1),
            None,
            None,
            None,
            None,
            None,
        )
        state = jax.random.uniform(self.key, self.state_shape)

        computed_best_action = q.best_action(q.params, state, self.key)

        # -1 since head behavioral policy equals to [0, ..., 0, 1]
        best_action = jnp.argmax(q(q.params, state)[0, -1]).astype(jnp.int8)
        self.assertEqual(best_action, computed_best_action)

    def test_update_heads(self) -> None:
        q = AtariiDQN(
            self.importance_iteration,
            self.state_shape,
            self.n_actions,
            self.gamma,
            self.key,
            self.head_behaviorial_probability,
            None,
            None,
            None,
            None,
            None,
        )
        state = jax.random.uniform(self.key, (50,) + self.state_shape)

        output = q(q.params, state)

        q.params = q.update_heads(q.params)

        forward_output = q(q.params, state)

        self.assertAlmostEqual(np.linalg.norm(forward_output[:, :-1] - output[:, 1:]), 0, places=8)
