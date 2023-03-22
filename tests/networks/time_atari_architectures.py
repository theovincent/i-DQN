from time import time
import jax
import jax.numpy as jnp
import numpy as np

from idqn.networks.q_architectures import AtariDQN


def run_cli():
    time_atari_q = TimeAtariDQN()

    time_atari_q.time_inference()
    time_atari_q.time_compute_target()
    time_atari_q.time_loss()
    time_atari_q.time_best_action()


class TimeAtariDQN:
    def __init__(self) -> None:
        self.random_seed = np.random.randint(1000)
        print(f"random seed {self.random_seed}")
        self.n_runs = 3000
        self.batch_size = 32
        self.key = jax.random.PRNGKey(self.random_seed)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.state_shape = (4, 84, 84)
        self.gamma = jax.random.uniform(self.key)

    def time_inference(self) -> None:
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None)
        state_key = self.key

        # Outside of the count: time to jit the __call__ function
        jax.block_until_ready(q(q.params, jax.random.uniform(state_key, self.state_shape)))

        t_begin = time()

        for _ in range(self.n_runs):
            state_key, key = jax.random.split(state_key)
            jax.block_until_ready(q(q.params, jax.random.uniform(key, self.state_shape)))

        print("Time inference: ", (time() - t_begin) / self.n_runs)

    def time_compute_target(self) -> None:
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None)
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

        jax.block_until_ready(q.compute_target(q.params, samples))

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

            jax.block_until_ready(q.compute_target(q.params, samples))

        print("Time compute target: ", (time() - t_begin) / self.n_runs)

    def time_loss(self) -> None:
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None)
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

        jax.block_until_ready(q.loss_and_grad(q.params, q.params, samples))

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

            jax.block_until_ready(q.loss_and_grad(q.params, q.params, samples))

        print("Time loss: ", (time() - t_begin) / self.n_runs)

    def time_best_action(self) -> None:
        q = AtariDQN(self.state_shape, self.n_actions, self.gamma, self.key, None, None, None)
        state_key = self.key

        # Outside of the count: time to jit the __call__ function
        jax.block_until_ready(q.best_action(None, q.params, jax.random.uniform(self.key, self.state_shape)))

        t_begin = time()

        for _ in range(self.n_runs):
            state_key, key = jax.random.split(state_key)
            jax.block_until_ready(q.best_action(None, q.params, jax.random.uniform(key, self.state_shape)))

        print("Time best action: ", (time() - t_begin) / self.n_runs)
