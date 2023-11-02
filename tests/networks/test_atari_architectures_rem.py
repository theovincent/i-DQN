import unittest
import jax
import jax.numpy as jnp
import numpy as np

from idqn.networks.architectures.rem import AtariREM
from tests.networks.utils import Generator
from idqn.sample_collection import IDX_RB


class TestAtariREM(unittest.TestCase):
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
        q = AtariREM(self.state_shape, self.n_actions, self.cumulative_gamma, self.key, None, None, None, None)

        state = self.generator.generate_state(self.key)
        state_copy = state.copy()

        output = q.apply(q.params, state)

        self.assertGreater(np.linalg.norm(output), 0)

        self.assertEqual(output.shape, (self.n_actions,))

        # test if the input has been changed
        self.assertEqual(np.linalg.norm(state - state_copy), 0)
        self.assertEqual(state.shape, state_copy.shape)

    def test_compute_target(self) -> None:
        q = AtariREM(self.state_shape, self.n_actions, self.cumulative_gamma, self.key, None, None, None, None)

        sample = self.generator.generate_sample(self.key)

        computed_target = q.compute_target(q.params, sample)

        target = sample[IDX_RB["reward"]] + (1 - sample[IDX_RB["terminal"]]) * self.cumulative_gamma * jnp.max(
            q.apply(q.params, sample[IDX_RB["next_state"]])
        )
        self.assertAlmostEqual(target, computed_target)

    def test_loss(self) -> None:
        q = AtariREM(self.state_shape, self.n_actions, self.cumulative_gamma, self.key, None, None, None, None)

        sample = self.generator.generate_sample(self.key)

        computed_loss = q.loss(q.params, q.params, sample, None)

        target = q.compute_target(q.params, sample)
        prediction = q.apply(q.params, sample[IDX_RB["state"]])[sample[IDX_RB["action"]].astype(jnp.int8)]
        loss = np.square(target - prediction)

        self.assertAlmostEqual(loss, computed_loss)

    def test_best_action(self) -> None:
        q = AtariREM(self.state_shape, self.n_actions, self.cumulative_gamma, self.key, None, None, None, None)

        state = self.generator.generate_state(self.key)

        computed_best_action = q.best_action(q.params, state, None)

        best_action = jnp.argmax(q.apply(q.params, state)).astype(jnp.int8)

        self.assertEqual(best_action, computed_best_action)
