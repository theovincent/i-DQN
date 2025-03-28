import unittest
import numpy as np
import jax
import jax.numpy as jnp

from slimdqn.networks.idqn import iDQN
from tests.utils import Generator


class TestDQN(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        self.key = jax.random.PRNGKey(self.random_seed)

        key_actions, key_n_networks, key_feature_1, key_feature_2, key_feature_3, key_feature_4 = jax.random.split(
            self.key, 6
        )
        self.observation_dim = (84, 84, 4)
        self.n_actions = int(jax.random.randint(key_actions, (), minval=2, maxval=10))
        self.n_networks = int(jax.random.randint(key_n_networks, (), 1, 10))
        self.q = iDQN(
            self.key,
            self.observation_dim,
            self.n_actions,
            self.n_networks,
            [
                jax.random.randint(key_feature_1, (), minval=1, maxval=10),
                jax.random.randint(key_feature_2, (), minval=1, maxval=10),
                jax.random.randint(key_feature_3, (), minval=1, maxval=10),
                jax.random.randint(key_feature_4, (), minval=1, maxval=10),
            ],
            "impala",
            0.001,
            0.94,
            1,
            1,
            1,
            1,
        )

        self.generator = Generator(None, self.observation_dim, self.n_actions)

    def test_compute_target(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        sample = self.generator.sample(self.key)
        idx_params = jax.random.randint(self.key, (), 0, self.n_networks)

        computed_target = self.q.compute_target(jax.tree.map(lambda param: param[idx_params], self.q.params), sample)

        next_q_values = self.q.network.apply(
            jax.tree.map(lambda param: param[idx_params], self.q.params), sample.next_state
        )
        target = sample.reward + (1 - sample.is_terminal) * self.q.gamma * jnp.max(next_q_values)

        self.assertEqual(next_q_values.shape, (self.n_actions,))
        self.assertEqual(target, computed_target)

    def test_loss(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        sample = self.generator.sample(self.key)
        idx_params = jax.random.randint(self.key, (), 0, self.n_networks)
        params = jax.tree.map(lambda param: param[idx_params], self.q.params)

        computed_loss = self.q.loss(params, params, sample)

        target = self.q.compute_target(params, sample)
        prediction = self.q.network.apply(params, sample.state)[sample.action]
        loss = np.square(target - prediction)

        self.assertEqual(loss, computed_loss)

    def test_best_action(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)

        computed_best_action = self.q.best_action(self.q.params, state, self.key)

        idx_params = jax.random.randint(self.key, (), 0, self.n_networks)
        q_values = self.q.network.apply(jax.tree.map(lambda param: param[idx_params], self.q.params), state)
        best_action = jnp.argmax(q_values)

        self.assertEqual(q_values.shape, (self.n_actions,))
        self.assertEqual(best_action, computed_best_action)
