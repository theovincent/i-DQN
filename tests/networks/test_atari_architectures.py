import unittest
import jax
import jax.numpy as jnp
import numpy as np

from idqn.sample_collection import IDX_RB
from idqn.networks.q_architectures import AtariDQN, AtariIQN, AtariiDQN, AtariiIQN


class TestAtariDQN(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        print(f"random seed {self.random_seed}")
        self.key = jax.random.PRNGKey(self.random_seed)
        self.state_shape = (84, 84, 4)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.gamma = jax.random.uniform(self.key)

    def test_output(self) -> None:
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)

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
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)

        rewards = jax.random.uniform(self.key, (10,), minval=-1, maxval=1)
        absorbings = jax.random.randint(self.key, (10,), 0, 2)
        next_states = jax.random.uniform(self.key, (10,) + self.state_shape, minval=-1, maxval=1)
        samples = (
            0,  # state
            0,  # action
            jnp.array(rewards, dtype=jnp.float32),  # reward
            jnp.array(next_states, dtype=jnp.float32),  # next_state
            0,  # next_action
            0,  # next_reward
            jnp.array(absorbings, dtype=jnp.bool_),  # terminal
        )

        computed_targets = q.compute_target(q.params, samples)

        for idx_sample in range(10):
            target = rewards[idx_sample] + (1 - absorbings[idx_sample]) * self.gamma * jnp.max(
                q.apply(q.params, next_states[idx_sample])
            )
            self.assertAlmostEqual(computed_targets[idx_sample], target, places=6)

    def test_loss(self) -> None:
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)

        states = jax.random.uniform(self.key, (10,) + self.state_shape, minval=-1, maxval=1)
        actions = jax.random.randint(self.key, (10,), minval=0, maxval=self.n_actions)
        key, _ = jax.random.split(self.key)
        rewards = jax.random.uniform(key, (10,), minval=-1, maxval=1)
        absorbings = jax.random.randint(key, (10,), 0, 2)
        next_states = jax.random.uniform(key, (10,) + self.state_shape, minval=-1, maxval=1)
        samples = (
            jnp.array(states, dtype=jnp.float32),  # state
            jnp.array(actions, dtype=jnp.int8),  # action
            jnp.array(rewards, dtype=jnp.float32),  # reward
            jnp.array(next_states, dtype=jnp.float32),  # next_state
            0,  # next_action
            0,  # next_reward
            jnp.array(absorbings, dtype=jnp.bool_),  # terminal
        )

        computed_loss = q.loss(q.params, q.params, samples)

        targets = q.compute_target(q.params, samples)
        predictions = np.zeros(10) * np.nan

        for idx_sample in range(10):
            predictions[idx_sample] = q.apply(q.params, states[idx_sample])[0, actions.astype(jnp.int8)[idx_sample]]

        self.assertAlmostEqual(computed_loss, np.square(targets - predictions).mean(), places=6)

    def test_random_action(self) -> None:
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)

        self.assertEqual(q.random_action(self.key).dtype, jnp.int8)

    def test_best_action(self) -> None:
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)

        state = jax.random.uniform(self.key, self.state_shape, minval=-1, maxval=1)

        computed_best_action = q.best_action(q.params, state, None)

        best_action = jnp.argmax(q.apply(q.params, state)[0]).astype(jnp.int8)
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
        absorbings = jax.random.randint(self.key, (10,), 0, 2)
        next_states = jax.random.uniform(self.key, (10,) + self.state_shape, minval=-1, maxval=1)
        samples = (
            0,  # state
            0,  # action
            jnp.array(rewards, dtype=jnp.float32),  # reward
            jnp.array(next_states, dtype=jnp.float32),  # next_state
            0,  # next_action
            0,  # next_reward
            jnp.array(absorbings, dtype=jnp.bool_),  # terminal
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
                + (1 - absorbings[idx_sample]) * self.gamma * quantiles_targets[idx_sample, :, action]
            )
            self.assertAlmostEqual(jnp.linalg.norm(computed_targets[idx_sample] - target), 0)

    def test_loss(self) -> None:
        q = AtariIQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None, None)
        q.n_quantiles = 13
        q.n_quantiles_target = 9

        states = jax.random.uniform(self.key, (10,) + self.state_shape, minval=-1, maxval=1)
        actions = jax.random.randint(self.key, (10,), minval=0, maxval=self.n_actions)
        key, _ = jax.random.split(self.key)
        rewards = jax.random.uniform(key, (10,), minval=-1, maxval=1)
        absorbings = jax.random.randint(key, (10,), 0, 2)
        next_states = jax.random.uniform(key, (10,) + self.state_shape, minval=-1, maxval=1)
        samples = (
            jnp.array(states, dtype=jnp.float32),  # state
            jnp.array(actions, dtype=jnp.int8),  # action
            jnp.array(rewards, dtype=jnp.float32),  # reward
            jnp.array(next_states, dtype=jnp.float32),  # next_state
            0,  # next_action
            0,  # next_reward
            jnp.array(absorbings, dtype=jnp.bool_),  # terminal
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

        computed_best_action = q.best_action(q.params, state, self.key)

        quantiles_policy, _ = q.network.apply(q.params, state, self.key, q.n_quantiles_policy)
        value_policy = jnp.mean(quantiles_policy, axis=1)[0]
        best_action = jnp.argmax(value_policy)

        self.assertEqual(best_action, computed_best_action)


class TestAtariiDQN(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        print(f"random seed {self.random_seed}")
        self.key = jax.random.PRNGKey(self.random_seed)
        self.n_heads = int(jax.random.randint(self.key, (), minval=2, maxval=50))
        self.state_shape = (4, 84, 84)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.gamma = jax.random.uniform(self.key)
        self.head_behaviorial_probability = jax.random.uniform(self.key, (self.n_heads,), minval=1, maxval=10)

    def test_output(self) -> None:
        q = AtariiDQN(
            self.n_heads,
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
            self.gamma,
            self.key,
            self.head_behaviorial_probability,
            None,
            None,
            None,
            None,
            None,
        )

        rewards = jax.random.uniform(self.key, (10,), minval=-1, maxval=1)
        absorbings = jax.random.randint(self.key, (10,), 0, 2)
        next_states = jax.random.uniform(self.key, (10,) + self.state_shape, minval=-1, maxval=1)
        samples = (
            0,  # state
            0,  # action
            jnp.array(rewards, dtype=jnp.float32),  # reward
            jnp.array(next_states, dtype=jnp.float32),  # next_state
            0,  # next_action
            0,  # next_reward
            jnp.array(absorbings, dtype=jnp.bool_),  # terminal
            0,  # indices
        )

        computed_targets = q.compute_target(q.params, samples)

        for idx_sample in range(10):
            for idx_head in range(self.n_heads):
                target = rewards[idx_sample] + (1 - absorbings[idx_sample]) * self.gamma * jnp.max(
                    q.apply(q.params, next_states[idx_sample])[:, idx_head]
                )
                self.assertAlmostEqual(computed_targets[idx_sample, idx_head], target, places=6)

    def test_loss(self) -> None:
        q = AtariiDQN(
            self.n_heads,
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
        states = jax.random.uniform(self.key, (10,) + self.state_shape, minval=-1, maxval=1)
        actions = jax.random.randint(self.key, (10,), minval=0, maxval=self.n_actions)
        key, _ = jax.random.split(self.key)
        rewards = jax.random.uniform(key, (10,), minval=-1, maxval=1)
        absorbings = jax.random.randint(key, (10,), 0, 2)
        next_states = jax.random.uniform(key, (10,) + self.state_shape, minval=-1, maxval=1)
        samples = (
            jnp.array(states, dtype=jnp.float32),  # state
            jnp.array(actions, dtype=jnp.int8),  # action
            jnp.array(rewards, dtype=jnp.float32),  # reward
            jnp.array(next_states, dtype=jnp.float32),  # next_state
            0,  # next_action
            0,  # next_reward
            jnp.array(absorbings, dtype=jnp.bool_),  # terminal
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
            self.gamma,
            self.key,
            jnp.zeros(self.n_heads).at[-1].set(1),
            None,
            None,
            None,
            None,
            None,
        )
        state = jax.random.uniform(self.key, self.state_shape, minval=-1, maxval=1)

        computed_best_action = q.best_action(q.params, state, self.key)

        # -1 since head behavioral policy equals to [0, ..., 0, 1]
        best_action = jnp.argmax(q.apply(q.params, state)[0, -1]).astype(jnp.int8)
        self.assertEqual(best_action, computed_best_action)

    def test_update_heads(self) -> None:
        q = AtariiDQN(
            self.n_heads,
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
        state = jax.random.uniform(self.key, (50,) + self.state_shape, minval=-1, maxval=1)

        output = q.apply(q.params, state)

        q.params = q.update_heads(q.params)

        forward_output = q.apply(q.params, state)

        self.assertAlmostEqual(np.linalg.norm(forward_output[:, :-1] - output[:, 1:]), 0, places=8)


class TestAtariiIQN(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        print(f"random seed {self.random_seed}")
        self.key = jax.random.PRNGKey(self.random_seed)
        self.n_heads = int(jax.random.randint(self.key, (), minval=2, maxval=50))
        self.state_shape = (4, 84, 84)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.gamma = jax.random.uniform(self.key)
        self.head_behaviorial_probability = jax.random.uniform(self.key, (self.n_heads,), minval=1, maxval=10)

    def test_output(self) -> None:
        q = AtariiIQN(
            self.n_heads,
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
            32,
            64,
            64,
        )

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

        self.assertEqual(output.shape, (1, self.n_heads - 1, q.n_quantiles, self.n_actions))
        self.assertEqual(
            output_batch.shape, (50, self.n_heads, q.n_quantiles_policy + q.n_quantiles_target, self.n_actions)
        )

        # test if the input has been changed
        self.assertEqual(np.linalg.norm(state - state_copy), 0)
        self.assertEqual(state.shape, state_copy.shape)

    def test_compute_target(self) -> None:
        q = AtariiIQN(
            self.n_heads,
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
            32,
            64,
            64,
        )

        rewards = jax.random.uniform(self.key, (10,), minval=-1, maxval=1)
        absorbings = jax.random.randint(self.key, (10,), 0, 2)
        next_states = jax.random.uniform(self.key, (10,) + self.state_shape, minval=-1, maxval=1)
        samples = (
            0,  # state
            0,  # action
            jnp.array(rewards, dtype=jnp.float32),  # reward
            jnp.array(next_states, dtype=jnp.float32),  # next_state
            0,  # next_action
            0,  # next_reward
            jnp.array(absorbings, dtype=jnp.bool_),  # terminal
            0,  # indices
            jax.random.split(self.key)[0],  # key
            jax.random.split(self.key)[1],  # next_key
        )

        computed_targets = q.compute_target(q.params, samples)

        quantiles_policy_targets, _ = q.network.apply(
            q.target_params, next_states, samples[IDX_RB["next_key"]], q.n_quantiles_policy + q.n_quantiles_target
        )
        quantiles_policy, quantiles_targets = (
            quantiles_policy_targets[:, :, : q.n_quantiles_policy],
            quantiles_policy_targets[:, :, q.n_quantiles_policy :],
        )

        for idx_sample in range(10):
            for idx_head in range(self.n_heads):
                value_policy = jnp.mean(quantiles_policy[idx_sample, idx_head], axis=0)
                action = jnp.argmax(value_policy)

                target = (
                    rewards[idx_sample]
                    + (1 - absorbings[idx_sample]) * self.gamma * quantiles_targets[idx_sample, idx_head, :, action]
                )
                self.assertAlmostEqual(jnp.linalg.norm(computed_targets[idx_sample, idx_head] - target), 0, places=6)

    def test_loss(self) -> None:
        n_heads = 5
        q = AtariiIQN(
            n_heads,
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
            32,
            13,
            9,
        )

        states = jax.random.uniform(self.key, (10,) + self.state_shape, minval=-1, maxval=1)
        actions = jax.random.randint(self.key, (10,), minval=0, maxval=self.n_actions)
        key, _ = jax.random.split(self.key)
        rewards = jax.random.uniform(key, (10,), minval=-1, maxval=1)
        absorbings = jax.random.randint(key, (10,), 0, 2)
        next_states = jax.random.uniform(key, (10,) + self.state_shape, minval=-1, maxval=1)
        samples = (
            jnp.array(states, dtype=jnp.float32),  # state
            jnp.array(actions, dtype=jnp.int8),  # action
            jnp.array(rewards, dtype=jnp.float32),  # reward
            jnp.array(next_states, dtype=jnp.float32),  # next_state
            0,  # next_action
            0,  # next_reward
            jnp.array(absorbings, dtype=jnp.bool_),  # terminal
            jax.random.split(self.key)[0],  # key
            jax.random.split(self.key)[1],  # next_key
        )
        computed_loss = q.loss(q.params, q.params, samples)

        targets = q.compute_target(q.params, samples)
        predictions, quantiles = q.network.apply(q.params, states, samples[IDX_RB["key"]], q.n_quantiles)

        loss = 0

        for idx_sample in range(10):
            for idx_head in range(n_heads - 1):
                for idx_quantile in range(q.n_quantiles):
                    for idx_quantile_target in range(q.n_quantiles_target):
                        bellman_error = (
                            targets[idx_sample, idx_head, idx_quantile_target]
                            - predictions[idx_sample, idx_head + 1, idx_quantile, actions[idx_sample]]
                        )
                        huber_loss = (
                            1 / 2 * bellman_error**2 if jnp.abs(bellman_error) < 1 else jnp.abs(bellman_error) - 1 / 2
                        )
                        loss += jnp.abs(quantiles[idx_sample, idx_quantile] - (bellman_error < 0)) * huber_loss

        loss /= 10 * (n_heads - 1) * q.n_quantiles_target

        self.assertAlmostEqual(computed_loss, loss, places=5)

    def test_best_action(self) -> None:
        q = AtariiIQN(
            self.n_heads,
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
            32,
            64,
            64,
        )
        state = jax.random.uniform(self.key, self.state_shape, minval=-1, maxval=1)

        computed_best_action = q.best_action(q.params, state, self.key)

        quantiles_policy, _ = q.network.apply(q.params, state, self.key, q.n_quantiles_policy)
        # -1 since head behavioral policy equals to [0, ..., 0, 1]
        value_policy = jnp.mean(quantiles_policy[0, -1], axis=0)
        best_action = jnp.argmax(value_policy).astype(jnp.int8)

        self.assertEqual(best_action, computed_best_action)

    def test_update_heads(self) -> None:
        q = AtariiIQN(
            self.n_heads,
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
            32,
            64,
            64,
        )
        state = jax.random.uniform(self.key, (50,) + self.state_shape, minval=-1, maxval=1)

        output, _ = q.apply_n_quantiles(q.params, state, self.key)

        q.params = q.update_heads(q.params)

        forward_output, _ = q.apply_n_quantiles(q.params, state, self.key)

        self.assertAlmostEqual(np.linalg.norm(forward_output[:, :-1] - output[:, 1:]), 0, places=8)
