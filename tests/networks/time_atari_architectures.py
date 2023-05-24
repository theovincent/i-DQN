from time import time
import jax
import jax.numpy as jnp
import numpy as np

from idqn.networks.base_q import BaseQ
from idqn.networks.q_architectures import AtariDQN, AtariiDQN


def run_cli():
    print("Time DQN")
    time_atari_q = TimeAtariDQN()

    time_atari_q.time_inference()
    time_atari_q.time_compute_target()
    time_atari_q.time_loss()
    time_atari_q.time_best_action()

    print("\n\nTime iDQN")
    time_atari_iq = TimeAtariiDQN()

    time_atari_iq.time_inference()
    time_atari_iq.time_compute_target()
    time_atari_iq.time_loss()
    time_atari_iq.time_best_action()


class TimeAtariQ:
    def __init__(self, q: BaseQ) -> None:
        self.n_runs = 3000
        self.batch_size = 32
        self.q = q
        self.key = q.network_key
        self.state_shape = q.state_shape
        self.n_actions = q.n_actions

    def time_inference(self) -> None:
        state_key = self.key

        # Outside of the count: time to jit the __call__ function
        jax.block_until_ready(self.q(self.q.params, jax.random.uniform(state_key, self.state_shape)))

        t_begin = time()

        for _ in range(self.n_runs):
            state_key, key = jax.random.split(state_key)
            jax.block_until_ready(self.q(self.q.params, jax.random.uniform(key, self.state_shape)))

        print("Time inference: ", (time() - t_begin) / self.n_runs)

    def time_compute_target(self) -> None:
        batch_key = self.key

        # Outside of the count: time to jit the __call__ function
        rewards = jax.random.uniform(batch_key, (self.batch_size,))
        absorbings = jax.random.randint(batch_key, (self.batch_size,), 0, 2)
        next_states = jax.random.uniform(batch_key, (self.batch_size,) + self.state_shape)
        samples = {
            "reward": jnp.array(rewards, dtype=jnp.float32),
            "next_state": jnp.array(next_states, dtype=jnp.float32),
            "absorbing": jnp.array(absorbings, dtype=jnp.bool_),
        }

        jax.block_until_ready(self.q.compute_target(self.q.params, samples))

        t_begin = time()

        for _ in range(self.n_runs):
            batch_key, key = jax.random.split(batch_key)

            rewards = jax.random.uniform(key, (self.batch_size,))
            absorbings = jax.random.randint(key, (self.batch_size,), 0, 2)
            next_states = jax.random.uniform(key, (self.batch_size,) + self.state_shape)
            samples = {
                "reward": jnp.array(rewards, dtype=jnp.float32),
                "next_state": jnp.array(next_states, dtype=jnp.float32),
                "absorbing": jnp.array(absorbings, dtype=jnp.bool_),
            }

            jax.block_until_ready(self.q.compute_target(self.q.params, samples))

        print("Time compute target: ", (time() - t_begin) / self.n_runs)

    def time_loss(self) -> None:
        batch_key = self.key

        # Outside of the count: time to jit the __call__ function
        states = jax.random.uniform(batch_key, (self.batch_size,) + self.state_shape)
        actions = jax.random.uniform(batch_key, (self.batch_size,))
        batch_key, key = jax.random.split(batch_key)
        rewards = jax.random.uniform(key, (self.batch_size,))
        absorbings = jax.random.randint(key, (self.batch_size,), 0, 2)
        next_states = jax.random.uniform(key, (self.batch_size,) + self.state_shape)
        samples = {
            "state": jnp.array(states, dtype=jnp.float32),
            "action": jnp.array(actions, dtype=jnp.int8),
            "reward": jnp.array(rewards, dtype=jnp.float32),
            "next_state": jnp.array(next_states, dtype=jnp.float32),
            "absorbing": jnp.array(absorbings, dtype=jnp.bool_),
        }

        jax.block_until_ready(self.q.loss_and_grad(self.q.params, self.q.params, samples))

        t_begin = time()

        for _ in range(self.n_runs):
            batch_key, key = jax.random.split(batch_key)
            states = jax.random.uniform(key, (self.batch_size,) + self.state_shape)
            actions = jax.random.uniform(key, (self.batch_size,))
            batch_key, key = jax.random.split(batch_key)
            rewards = jax.random.uniform(key, (self.batch_size,))
            absorbings = jax.random.randint(key, (self.batch_size,), 0, 2)
            next_states = jax.random.uniform(key, (self.batch_size,) + self.state_shape)
            samples = {
                "state": jnp.array(states, dtype=jnp.float32),
                "action": jnp.array(actions, dtype=jnp.int8),
                "reward": jnp.array(rewards, dtype=jnp.float32),
                "next_state": jnp.array(next_states, dtype=jnp.float32),
                "absorbing": jnp.array(absorbings, dtype=jnp.bool_),
            }

            jax.block_until_ready(self.q.loss_and_grad(self.q.params, self.q.params, samples))

        print("Time loss: ", (time() - t_begin) / self.n_runs)

    def time_best_action(self) -> None:
        state_key = self.key

        # Outside of the count: time to jit the __call__ function
        # several time to jit all the underlying functions of q.best_action
        for _ in range(self.n_runs):
            state_key, key = jax.random.split(state_key)
            jax.block_until_ready(self.q.best_action(key, self.q.params, jax.random.uniform(key, self.state_shape)))

        t_begin = time()

        for _ in range(self.n_runs):
            state_key, key = jax.random.split(state_key)
            jax.block_until_ready(self.q.best_action(key, self.q.params, jax.random.uniform(key, self.state_shape)))

        print("Time best action: ", (time() - t_begin) / self.n_runs)


class TimeAtariDQN(TimeAtariQ):
    def __init__(self) -> None:
        self.random_seed = np.random.randint(1000)
        print(f"random seed {self.random_seed}")
        self.key = jax.random.PRNGKey(self.random_seed)
        self.state_shape = (4, 84, 84)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.gamma = jax.random.uniform(self.key)
        super().__init__(AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None))


class TimeAtariiDQN(TimeAtariQ):
    def __init__(self) -> None:
        self.random_seed = np.random.randint(1000)
        print(f"random seed {self.random_seed}", end=" ")
        self.key = jax.random.PRNGKey(self.random_seed)
        self.n_heads = jax.random.randint(self.key, (), minval=5, maxval=20)
        print(f"{self.n_heads} heads")
        self.importance_iteration = jax.random.uniform(self.key, (self.n_heads - 1,), minval=1, maxval=10)
        self.state_shape = (4, 84, 84)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.gamma = jax.random.uniform(self.key)
        self.head_behaviorial_probability = jax.random.uniform(self.key, (self.n_heads,), minval=1, maxval=10)
        super().__init__(
            q=AtariiDQN(
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
        )
