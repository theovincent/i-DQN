import os
import unittest
import jax
import numpy as np

from idqn.networks.learnable_multi_head_q import AtariMultiQ


class TestAtariEnv(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        self.key = jax.random.PRNGKey(self.random_seed)
        self.n_heads = int(jax.random.randint(self.key, (), minval=1, maxval=50) + 1)
        self.importance_iteration = jax.random.uniform(self.key, (self.n_heads,), minval=1, maxval=10)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.state_shape = (4, 84, 84)
        self.gamma = jax.random.uniform(self.key)
        self.n_shared_layers = int(jax.random.randint(self.key, (), minval=0, maxval=5))
        self.zero_initializer = bool(jax.random.randint(self.key, (), minval=0, maxval=2))

    def test_output(self) -> None:
        q = AtariMultiQ(
            self.importance_iteration,
            self.state_shape,
            self.n_actions,
            self.gamma,
            self.key,
            self.n_shared_layers,
            self.zero_initializer,
            None,
        )

        output = q(q.params, jax.random.uniform(self.key, self.state_shape))
        output_batch = q(q.params, jax.random.uniform(self.key, (50,) + self.state_shape))

        if self.zero_initializer:
            self.assertEqual(np.linalg.norm(output), 0, f"random seed {self.random_seed}")
            self.assertEqual(np.linalg.norm(output_batch), 0, f"random seed {self.random_seed}")
        else:
            self.assertGreater(np.linalg.norm(output), 0, f"random seed {self.random_seed}")
            self.assertGreater(np.linalg.norm(output_batch), 0, f"random seed {self.random_seed}")

        self.assertEqual(output.shape, (1, self.n_heads + 1, self.n_actions), f"random seed {self.random_seed}")
        self.assertEqual(output_batch.shape, (50, self.n_heads + 1, self.n_actions), f"random seed {self.random_seed}")

    def test_n_share_layers(self) -> None:
        q = AtariMultiQ(
            self.importance_iteration,
            self.state_shape,
            self.n_actions,
            self.gamma,
            self.key,
            self.n_shared_layers,
            self.zero_initializer,
            None,
        )

        if self.n_shared_layers > 0:
            self.assertIn(
                f"AtariNet/~/shared_first_head_layer_{self.n_shared_layers - 1}",
                q.params.keys(),
                f"random seed {self.random_seed}",
            )
            self.assertIn(
                f"AtariNet/~/shared_other_heads_layer_{self.n_shared_layers - 1}",
                q.params.keys(),
                f"random seed {self.random_seed}",
            )

        self.assertIn(
            f"AtariNet/~/head_{self.n_heads - 1}_layer_{self.n_shared_layers}",
            q.params.keys(),
            f"random seed {self.random_seed}",
        )
        self.assertIn(
            f"AtariNet/~/head_{self.n_heads - 1}_layer_{q.n_layers - 1}",
            q.params.keys(),
            f"random seed {self.random_seed}",
        )
        self.assertIn(
            f"AtariNet/~/head_0_layer_{self.n_shared_layers}", q.params.keys(), f"random seed {self.random_seed}"
        )
        self.assertIn(f"AtariNet/~/head_0_layer_{q.n_layers - 1}", q.params.keys(), f"random seed {self.random_seed}")

    def test_move_forward(self) -> None:
        q = AtariMultiQ(
            self.importance_iteration,
            self.state_shape,
            self.n_actions,
            self.gamma,
            self.key,
            self.n_shared_layers,
            zero_initializer=False,
            learning_rate=None,
        )
        state = jax.random.uniform(self.key, (50,) + self.state_shape)

        output = q(q.params, state)

        q.params = q.move_forward(q.params)

        forward_output = q(q.params, state)

        self.assertEqual(np.linalg.norm(forward_output[:, 0] - output[:, -1]), 0, f"random seed {self.random_seed}")
