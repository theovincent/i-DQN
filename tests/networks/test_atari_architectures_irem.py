import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict

from idqn.networks.architectures.idqn import AtariiDQN
from idqn.networks.architectures.irem import AtariiREM
from tests.networks.utils import Generator, assertArray
from idqn.sample_collection import IDX_RB


class TestAtariiREM(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        print(f"random seed {self.random_seed}")
        self.key = jax.random.PRNGKey(self.random_seed)
        self.n_heads = int(jax.random.randint(self.key, (), minval=2, maxval=10))
        self.state_shape = (84, 84, 4)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=15))
        self.cumulative_gamma = jax.random.uniform(self.key)
        self.head_behaviorial_probability = jax.random.uniform(self.key, (self.n_heads,), minval=1, maxval=10)
        self.generator = Generator(None, self.state_shape)

    def test_output(self) -> None:
        q = AtariiREM(
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

        state = self.generator.generate_state(self.key)
        state_copy = state.copy()

        output = q.apply(q.params, state)

        self.assertGreater(np.linalg.norm(output), 0)

        self.assertEqual(output.shape, (self.n_heads, self.n_actions))

        # test if the input has been changed
        self.assertEqual(np.linalg.norm(state - state_copy), 0)
        self.assertEqual(state.shape, state_copy.shape)

    def test_combiner(self) -> None:
        q = AtariiREM(
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
        unfrozen_params = q.params.unfreeze()
        combination = unfrozen_params["params_combiner"]["params"]["kernel"]

        self.assertGreaterEqual(np.min(combination), 0)
        self.assertGreaterEqual(1, np.max(combination))
        self.assertAlmostEqual(np.sum(combination), 1, places=6)

        unfrozen_params["params_combiner"]["params"]["kernel"] = jnp.zeros_like(combination)
        unfrozen_params["params_combiner"]["params"]["kernel"] = (
            unfrozen_params["params_combiner"]["params"]["kernel"].at[0, 0].set(1)
        )

        q.params = FrozenDict(unfrozen_params)

        q_idqn = AtariiDQN(
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

        state = self.generator.generate_state(self.key)

        output_dqn = q_idqn.apply(
            jax.tree_util.tree_map(lambda params: params[0], q.params["params_nets"]),
            state,
        )
        output = q.apply(q.params, state)

        assertArray(
            self.assertAlmostEqual,
            output_dqn,
            output,
            delta=np.abs(output_dqn.min()) / 1e3,
        )

    def test_compute_target(self) -> None:
        q = AtariiREM(
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

        sample = self.generator.generate_sample(self.key)

        computed_targets = q.compute_target(q.params, sample)

        targets = sample[IDX_RB["reward"]] + (1 - sample[IDX_RB["terminal"]]) * self.cumulative_gamma * jnp.max(
            q.apply(q.params, sample[IDX_RB["next_state"]])[:-1], axis=1
        )
        assertArray(self.assertAlmostEqual, targets, computed_targets)

    def test_loss(self) -> None:
        q = AtariiREM(
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
        sample = self.generator.generate_sample(self.key)

        computed_loss = q.loss(q.params, q.params, sample, None)

        targets = q.compute_target(q.params, sample)
        predictions = q.apply(q.params, sample[IDX_RB["state"]])[1:, sample[IDX_RB["action"]].astype(jnp.int8)]
        loss = np.square(targets - predictions).mean()

        self.assertAlmostEqual(loss, computed_loss, delta=abs(computed_loss) / 1e6)

    def test_best_action(self) -> None:
        q = AtariiREM(
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
        state = self.generator.generate_state(self.key)

        computed_best_action = q.best_action(q.params, state, self.key)

        # -1 since head behavioral policy equals to [0, ..., 0, 1]
        best_action = jnp.argmax(q.apply(q.params, state)[-1]).astype(jnp.int8)
        self.assertEqual(best_action, computed_best_action)

    def test_rolling_step(self) -> None:
        q = AtariiREM(
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
        state = self.generator.generate_state(self.key)

        output = q.apply(q.params, state)

        q.params = q.rolling_step(q.params)

        forward_output = q.apply(q.params, state)

        assertArray(self.assertAlmostEqual, forward_output[:-1], output[1:])
