import os
import unittest
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
import numpy as np

from idqn.networks.q_architectures import AtariDQN, AtariiDQN


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
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None)

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
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None)

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
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None)

        states = jax.random.uniform(self.key, (10,) + self.state_shape)
        actions = jax.random.uniform(self.key, (10,))
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
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None)

        self.assertEqual(q.random_action(self.key).dtype, jnp.int8)

    def test_best_action(self) -> None:
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None)

        state = jax.random.uniform(self.key, self.state_shape)

        computed_best_action = q.best_action(None, q.params, state)

        best_action = jnp.argmax(q(q.params, state)[0]).astype(jnp.int8)
        self.assertEqual(best_action, computed_best_action)

    def test_save_load(self) -> None:
        q_to_save = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, 1, None, None)
        q_to_save.save("tests/Q")

        key, _ = jax.random.split(self.key)
        q_to_load = AtariDQN(self.state_shape, self.n_actions, self.gamma, key, 1, None, None)
        q_to_load.load("tests/Q")

        def check_null(frozen_dict):
            if isinstance(frozen_dict, FrozenDict):
                for values in frozen_dict.values():
                    check_null(values)
            else:
                self.assertEqual(frozen_dict, 0)

        diff_params = jax.tree_util.tree_map(lambda x, y: jnp.linalg.norm(x - y), q_to_save.params, q_to_load.params)
        diff_target_params = jax.tree_util.tree_map(
            lambda x, y: jnp.linalg.norm(x - y), q_to_save.target_params, q_to_load.target_params
        )

        check_null(diff_params)
        check_null(diff_target_params)

        os.remove("tests/Q_online_params")
        os.remove("tests/Q_target_params")
        os.remove("tests/Q_optimizer")


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
        )
        states = jax.random.uniform(self.key, (10,) + self.state_shape)
        actions = jax.random.uniform(self.key, (10,))
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
        )
        state = jax.random.uniform(self.key, self.state_shape)

        computed_best_action = q.best_action(self.key, q.params, state)

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
        )
        state = jax.random.uniform(self.key, (50,) + self.state_shape)

        output = q(q.params, state)

        q.params = q.update_heads(q.params)

        forward_output = q(q.params, state)

        self.assertAlmostEqual(np.linalg.norm(forward_output[:, :-1] - output[:, 1:]), 0, places=8)
