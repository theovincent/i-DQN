import unittest
import jax
import jax.numpy as jnp
import numpy as np

from idqn.networks.architectures.iqn import AtariIQN
from tests.networks.utils import Generator, assertArray
from idqn.sample_collection import IDX_RB


class TestAtariIQN(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        print(f"random seed {self.random_seed}")
        self.key = jax.random.PRNGKey(self.random_seed)
        self.state_shape = (84, 84, 4)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.cumulative_gamma = jax.random.uniform(self.key)
        self.generator = Generator(None, self.state_shape)

    def test_output(self) -> None:
        q = AtariIQN(self.state_shape, self.n_actions, self.cumulative_gamma, self.key, None, None, None, None)

        state = self.generator.generate_state(self.key)
        state_copy = state.copy()

        output, quantiles = q.apply(q.params, state, self.key, q.n_quantiles)

        self.assertGreater(np.linalg.norm(output), 0)

        self.assertEqual(quantiles.shape, (q.n_quantiles,))

        self.assertEqual(output.shape, (q.n_quantiles, self.n_actions))

        # test if the input has been changed
        self.assertEqual(np.linalg.norm(state - state_copy), 0)
        self.assertEqual(state.shape, state_copy.shape)

    def test_compute_target(self) -> None:
        q = AtariIQN(self.state_shape, self.n_actions, self.cumulative_gamma, self.key, None, None, None, None)

        sample = self.generator.generate_sample(self.key)

        # output (n_quantiles_target)
        computed_targets = q.compute_target(q.target_params, sample, self.key)

        # output (n_quantiles_policy + n_quantiles_target, n_actions)
        quantiles_policy_target, _ = q.apply(
            q.target_params, sample[IDX_RB["next_state"]], self.key, q.n_quantiles_policy + q.n_quantiles_target
        )
        quantiles_policy = quantiles_policy_target[: q.n_quantiles_policy]
        quantiles_targets = quantiles_policy_target[q.n_quantiles_policy :]

        # output (n_actions)
        value_policy = jnp.mean(quantiles_policy, axis=0)
        action = jnp.argmax(value_policy)

        targets = (
            sample[IDX_RB["reward"]]
            + (1 - sample[IDX_RB["terminal"]]) * self.cumulative_gamma * quantiles_targets[:, action]
        )

        assertArray(self.assertAlmostEqual, targets, computed_targets)

    def test_loss(self) -> None:
        q = AtariIQN(self.state_shape, self.n_actions, self.cumulative_gamma, self.key, None, None, None, None)

        sample = self.generator.generate_sample(self.key)

        computed_loss = q.loss(q.params, q.params, sample, self.key)

        # output (n_quantiles_target)
        targets = q.compute_target(q.params, sample, jax.random.split(self.key)[1])
        # output (n_quantiles, n_actions)
        predictions_values, quantiles = q.apply(q.params, sample[IDX_RB["state"]], self.key, q.n_quantiles)
        # output (n_quantiles)
        predictions = predictions_values[:, sample[IDX_RB["action"]]]

        loss = 0

        for idx_quantile in range(q.n_quantiles):
            for idx_quantile_target in range(q.n_quantiles_target):
                bellman_error = targets[idx_quantile_target] - predictions[idx_quantile]
                huber_loss = (
                    1 / 2 * bellman_error**2 if jnp.abs(bellman_error) <= 1 else jnp.abs(bellman_error) - 1 / 2
                )
                loss += jnp.abs(quantiles[idx_quantile] - (bellman_error < 0).astype(jnp.float32)) * huber_loss

        loss /= q.n_quantiles_target

        self.assertAlmostEqual(loss, computed_loss, delta=loss / 1e5)

    def test_best_action(self) -> None:
        q = AtariIQN(self.state_shape, self.n_actions, self.cumulative_gamma, self.key, None, None, None, None)

        state = self.generator.generate_state(self.key)

        computed_best_action = q.best_action(q.params, state, self.key)

        quantiles_policy, _ = q.network.apply(q.params, state, self.key, q.n_quantiles_policy)
        value_policy = jnp.mean(quantiles_policy, axis=0)
        best_action = jnp.argmax(value_policy)

        self.assertEqual(best_action, computed_best_action)
