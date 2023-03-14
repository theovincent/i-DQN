import unittest
import jax
import jax.numpy as jnp
import numpy as np

from idqn.networks.q_architectures import AtariDQN, AtariiDQN


class TestAtariDQN(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = 10  # np.random.randint(1000)
        self.key = jax.random.PRNGKey(self.random_seed)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.state_shape = (4, 84, 84)
        self.gamma = jax.random.uniform(self.key)
        self.zero_initializer = bool(jax.random.randint(self.key, (), minval=0, maxval=2))

    def test_output(self) -> None:
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, self.zero_initializer, None)

        output = q(q.params, jax.random.uniform(self.key, self.state_shape))
        output_batch = q(q.params, jax.random.uniform(self.key, (50,) + self.state_shape))

        if self.zero_initializer:
            self.assertEqual(np.linalg.norm(output), 0, f"random seed {self.random_seed}")
            self.assertEqual(np.linalg.norm(output_batch), 0, f"random seed {self.random_seed}")
        else:
            self.assertGreater(np.linalg.norm(output), 0, f"random seed {self.random_seed}")
            self.assertGreater(np.linalg.norm(output_batch), 0, f"random seed {self.random_seed}")

        self.assertEqual(output.shape, (1, self.n_actions), f"random seed {self.random_seed}")
        self.assertEqual(output_batch.shape, (50, self.n_actions), f"random seed {self.random_seed}")

    def test_compute_target(self) -> None:
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, self.zero_initializer, None)

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
            self.assertAlmostEqual(computed_targets[idx_sample], target, msg=f"random seed {self.random_seed}")

    def test_loss(self) -> None:
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, self.zero_initializer, None)

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

        self.assertAlmostEqual(
            computed_loss, np.square(targets - predictions).mean(), msg=f"random seed {self.random_seed}"
        )

    def test_random_action(self) -> None:
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, self.zero_initializer, None)

        self.assertEqual(q.random_action(self.key).dtype, jnp.int8, msg=f"random seed {self.random_seed}")

    def test_best_action(self) -> None:
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, self.zero_initializer, None)

        state = jax.random.uniform(self.key, self.state_shape)

        computed_best_action = q.best_action(None, q.params, state)

        best_action = jnp.argmax(q(q.params, state)[0]).astype(jnp.int8)
        self.assertEqual(best_action, computed_best_action, msg=f"random seed {self.random_seed}")


# class TestAtariiDQN(unittest.TestCase):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.random_seed = np.random.randint(1000)
#         self.key = jax.random.PRNGKey(self.random_seed)
#         self.n_heads = int(jax.random.randint(self.key, (), minval=1, maxval=50) + 1)
#         self.importance_iteration = jax.random.uniform(self.key, (self.n_heads,), minval=1, maxval=10)
#         self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
#         self.state_shape = (4, 84, 84)
#         self.gamma = jax.random.uniform(self.key)
#         self.n_shared_layers = int(jax.random.randint(self.key, (), minval=0, maxval=5))
#         self.zero_initializer = bool(jax.random.randint(self.key, (), minval=0, maxval=2))

#     def test_output(self) -> None:
#         q = AtariiDQN(
#             self.importance_iteration,
#             self.state_shape,
#             self.n_actions,
#             self.gamma,
#             self.key,
#             self.n_shared_layers,
#             self.zero_initializer,
#             None,
#         )

#         output = q(q.params, jax.random.uniform(self.key, self.state_shape))
#         output_batch = q(q.params, jax.random.uniform(self.key, (50,) + self.state_shape))

#         if self.zero_initializer:
#             self.assertEqual(np.linalg.norm(output), 0, f"random seed {self.random_seed}")
#             self.assertEqual(np.linalg.norm(output_batch), 0, f"random seed {self.random_seed}")
#         else:
#             self.assertGreater(np.linalg.norm(output), 0, f"random seed {self.random_seed}")
#             self.assertGreater(np.linalg.norm(output_batch), 0, f"random seed {self.random_seed}")

#         self.assertEqual(output.shape, (1, self.n_heads + 1, self.n_actions), f"random seed {self.random_seed}")
#         self.assertEqual(output_batch.shape, (50, self.n_heads + 1, self.n_actions), f"random seed {self.random_seed}")

# def test_n_share_layers(self) -> None:
#     q = AtariiDQN(
#         self.importance_iteration,
#         self.state_shape,
#         self.n_actions,
#         self.gamma,
#         self.key,
#         self.n_shared_layers,
#         self.zero_initializer,
#         None,
#     )

#     if self.n_shared_layers > 0:
#         self.assertIn(
#             f"AtariNet/~/shared_first_head_layer_{self.n_shared_layers - 1}",
#             q.params.keys(),
#             f"random seed {self.random_seed}",
#         )
#         self.assertIn(
#             f"AtariNet/~/shared_other_heads_layer_{self.n_shared_layers - 1}",
#             q.params.keys(),
#             f"random seed {self.random_seed}",
#         )

#     self.assertIn(
#         f"AtariNet/~/head_{self.n_heads - 1}_layer_{self.n_shared_layers}",
#         q.params.keys(),
#         f"random seed {self.random_seed}",
#     )
#     self.assertIn(
#         f"AtariNet/~/head_{self.n_heads - 1}_layer_{q.n_layers - 1}",
#         q.params.keys(),
#         f"random seed {self.random_seed}",
#     )
#     self.assertIn(
#         f"AtariNet/~/head_0_layer_{self.n_shared_layers}", q.params.keys(), f"random seed {self.random_seed}"
#     )
#     self.assertIn(f"AtariNet/~/head_0_layer_{q.n_layers - 1}", q.params.keys(), f"random seed {self.random_seed}")

# def test_move_forward(self) -> None:
#     q = AtariiDQN(
#         self.importance_iteration,
#         self.state_shape,
#         self.n_actions,
#         self.gamma,
#         self.key,
#         self.n_shared_layers,
#         zero_initializer=False,
#         learning_rate=None,
#     )
#     state = jax.random.uniform(self.key, (50,) + self.state_shape)

#     output = q(q.params, state)

#     q.params = q.move_forward(q.params)

#     forward_output = q(q.params, state)

#     self.assertEqual(np.linalg.norm(forward_output[:, 0] - output[:, -1]), 0, f"random seed {self.random_seed}")
