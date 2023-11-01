from time import time
import jax
import jax.numpy as jnp
import numpy as np

from idqn.networks.base import BaseQ
from idqn.networks.architectures.dqn import AtariDQN
from idqn.networks.architectures.iqn import AtariIQN
from idqn.networks.architectures.rem import AtariREM
from idqn.networks.architectures.idqn import AtariiDQN
from idqn.networks.architectures.iiqn import AtariiIQN
from idqn.networks.architectures.irem import AtariiREM

RANDOM_SEED = 845  # np.random.randint(1000)


def run_cli():
    print("Time DQN")
    time_atari_dqn = TimeAtariDQN()

    time_atari_dqn.time_inference()
    time_atari_dqn.time_compute_target()
    time_atari_dqn.time_loss()
    time_atari_dqn.time_best_action()

    print("\n\nTime IQN")
    time_atari_iqn = TimeAtariIQN()

    time_atari_iqn.time_inference()
    time_atari_iqn.time_compute_target()
    time_atari_iqn.time_loss()
    time_atari_iqn.time_best_action()

    print("\n\nTime REM")
    time_atari_rem = TimeAtariREM()

    time_atari_rem.time_inference()
    time_atari_rem.time_compute_target()
    time_atari_rem.time_loss()
    time_atari_rem.time_best_action()

    print("\n\nTime iDQN")
    time_atari_idqn = TimeAtariiDQN()

    time_atari_idqn.time_inference()
    time_atari_idqn.time_compute_target()
    time_atari_idqn.time_loss()
    time_atari_idqn.time_best_action()

    print("\n\nTime iIQN")
    time_atari_iiqn = TimeAtariiIQN()

    time_atari_iiqn.time_inference()
    time_atari_iiqn.time_compute_target()
    time_atari_iiqn.time_loss()
    time_atari_iiqn.time_best_action()

    print("\n\nTime iREM")
    time_atari_rem = TimeAtariiREM()

    time_atari_rem.time_inference()
    time_atari_rem.time_compute_target()
    time_atari_rem.time_loss()
    time_atari_rem.time_best_action()


class TimeAtariQ:
    def __init__(self, q: BaseQ) -> None:
        self.n_runs = 3000
        self.batch_size = 32
        self.q = q
        self.key = q.network_key
        self.n_actions = q.n_actions

    def time_inference(self) -> None:
        state_key = self.key
        apply_func = jax.jit(self.q.apply)

        # Outside of the count: time to jit the __call__ function
        jax.block_until_ready(
            apply_func(self.q.params, jax.random.uniform(state_key, (self.batch_size,) + self.state_shape))
        )

        t_begin = time()

        for _ in range(self.n_runs):
            state_key, key = jax.random.split(state_key)
            jax.block_until_ready(
                apply_func(self.q.params, jax.random.uniform(key, (self.batch_size,) + self.state_shape))
            )

        print("Time inference: ", (time() - t_begin) / self.n_runs)

    def time_compute_target(self) -> None:
        batch_key = self.key
        compute_target_func = jax.jit(self.q.compute_target)

        # Outside of the count: time to jit the __call__ function
        rewards = jax.random.uniform(batch_key, (self.batch_size,))
        terminals = jax.random.randint(batch_key, (self.batch_size,), 0, 2)
        next_states = jax.random.uniform(batch_key, (self.batch_size,) + self.state_shape)
        samples = (
            0,  # state
            0,  # action
            jnp.array(rewards, dtype=jnp.float32),  # reward
            jnp.array(next_states, dtype=jnp.float32),  # next_state
            0,  # next_action
            0,  # next_reward
            jnp.array(terminals, dtype=jnp.bool_),  # terminal
            0,  # indices
        )
        samples = self.q.augment_samples(samples, key=self.q.network_key)

        jax.block_until_ready(compute_target_func(self.q.params, samples))

        t_begin = time()

        for _ in range(self.n_runs):
            batch_key, key = jax.random.split(batch_key)

            rewards = jax.random.uniform(key, (self.batch_size,))
            terminals = jax.random.randint(key, (self.batch_size,), 0, 2)
            next_states = jax.random.uniform(key, (self.batch_size,) + self.state_shape)
            samples = (
                0,  # state
                0,  # action
                jnp.array(rewards, dtype=jnp.float32),  # reward
                jnp.array(next_states, dtype=jnp.float32),  # next_state
                0,  # next_action
                0,  # next_reward
                jnp.array(terminals, dtype=jnp.bool_),  # terminal
                0,  # indices
            )
            samples = self.q.augment_samples(samples, key=key)

            jax.block_until_ready(compute_target_func(self.q.params, samples))

        print("Time compute target: ", (time() - t_begin) / self.n_runs)

    def time_loss(self) -> None:
        batch_key = self.key
        loss_and_grad_func = jax.jit(jax.value_and_grad(self.q.loss))

        # Outside of the count: time to jit the function
        states = jax.random.uniform(batch_key, (self.batch_size,) + self.state_shape)
        actions = jax.random.uniform(batch_key, (self.batch_size,))
        batch_key, key = jax.random.split(batch_key)
        rewards = jax.random.uniform(key, (self.batch_size,))
        terminals = jax.random.randint(key, (self.batch_size,), 0, 2)
        next_states = jax.random.uniform(key, (self.batch_size,) + self.state_shape)
        samples = (
            jnp.array(states, dtype=jnp.float32),  # state
            jnp.array(actions, dtype=jnp.int8),  # action
            jnp.array(rewards, dtype=jnp.float32),  # reward
            jnp.array(next_states, dtype=jnp.float32),  # next_state
            0,  # next_action
            0,  # next_reward
            jnp.array(terminals, dtype=jnp.bool_),  # terminal
            0,  # indices
        )
        samples = self.q.augment_samples(samples, key=self.q.network_key)

        jax.block_until_ready(loss_and_grad_func(self.q.params, self.q.params, samples))

        t_begin = time()

        for _ in range(self.n_runs):
            batch_key, key = jax.random.split(batch_key)
            states = jax.random.uniform(key, (self.batch_size,) + self.state_shape)
            actions = jax.random.uniform(key, (self.batch_size,))
            batch_key, key = jax.random.split(batch_key)
            rewards = jax.random.uniform(key, (self.batch_size,))
            terminals = jax.random.randint(key, (self.batch_size,), 0, 2)
            next_states = jax.random.uniform(key, (self.batch_size,) + self.state_shape)
            samples = (
                jnp.array(states, dtype=jnp.float32),  # state
                jnp.array(actions, dtype=jnp.int8),  # action
                jnp.array(rewards, dtype=jnp.float32),  # reward
                jnp.array(next_states, dtype=jnp.float32),  # next_state
                0,  # next_action
                0,  # next_reward
                jnp.array(terminals, dtype=jnp.bool_),  # terminal
                0,  # indices
            )
            samples = self.q.augment_samples(samples, key=key)

            jax.block_until_ready(loss_and_grad_func(self.q.params, self.q.params, samples))

        print("Time loss: ", (time() - t_begin) / self.n_runs)

    def time_best_action(self) -> None:
        state_key = self.key

        # Outside of the count: time to jit the function
        jax.block_until_ready(
            self.q.best_action(self.q.params, jax.random.uniform(state_key, self.state_shape), key=state_key)
        )

        t_begin = time()

        for _ in range(self.n_runs):
            state_key, key = jax.random.split(state_key)
            jax.block_until_ready(self.q.best_action(self.q.params, jax.random.uniform(key, self.state_shape), key=key))

        print("Time best action: ", (time() - t_begin) / self.n_runs)


class TimeAtariDQN(TimeAtariQ):
    def __init__(self) -> None:
        self.random_seed = RANDOM_SEED
        print(f"random seed {self.random_seed}")
        self.key = jax.random.PRNGKey(self.random_seed)
        self.state_shape = (84, 84, 4)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.cumulative_gamma = jax.random.uniform(self.key)
        super().__init__(
            AtariDQN(self.state_shape, self.n_actions, self.cumulative_gamma, self.key, None, None, None, None)
        )


class TimeAtariIQN(TimeAtariQ):
    def __init__(self) -> None:
        self.random_seed = RANDOM_SEED
        print(f"random seed {self.random_seed}")
        self.key = jax.random.PRNGKey(self.random_seed)
        self.state_shape = (84, 84, 4)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.cumulative_gamma = jax.random.uniform(self.key)
        super().__init__(
            AtariIQN(self.state_shape, self.n_actions, self.cumulative_gamma, self.key, None, None, None, None)
        )

    def time_inference(self) -> None:
        state_key = self.key
        apply_func = jax.jit(self.q.apply_n_quantiles)

        # Outside of the count: time to jit the __call__ function
        jax.block_until_ready(
            self.q.apply_n_quantiles(
                self.q.params, jax.random.uniform(state_key, (self.batch_size,) + self.state_shape), self.q.network_key
            )
        )

        t_begin = time()

        for _ in range(self.n_runs):
            state_key, key = jax.random.split(state_key)
            jax.block_until_ready(
                apply_func(
                    self.q.params, jax.random.uniform(key, (self.batch_size,) + self.state_shape), self.q.network_key
                )
            )

        print("Time inference: ", (time() - t_begin) / self.n_runs)


class TimeAtariREM(TimeAtariQ):
    def __init__(self) -> None:
        self.random_seed = RANDOM_SEED
        print(f"random seed {self.random_seed}")
        self.key = jax.random.PRNGKey(self.random_seed)
        self.state_shape = (84, 84, 4)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.cumulative_gamma = jax.random.uniform(self.key)
        super().__init__(
            AtariREM(self.state_shape, self.n_actions, self.cumulative_gamma, self.key, None, None, None, None)
        )


class TimeAtariiDQN(TimeAtariQ):
    def __init__(self) -> None:
        self.random_seed = RANDOM_SEED
        print(f"random seed {self.random_seed}", end=" ")
        self.key = jax.random.PRNGKey(self.random_seed)
        self.n_heads = int(jax.random.randint(self.key, (), minval=5, maxval=20))
        print(f"{self.n_heads} heads")
        self.state_shape = (84, 84, 4)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.cumulative_gamma = jax.random.uniform(self.key)
        self.head_behaviorial_probability = jax.random.uniform(self.key, (self.n_heads,), minval=1, maxval=10)
        shared_network = True
        print("Shared network" if shared_network else "Independant network")
        super().__init__(
            q=AtariiDQN(
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
                shared_network,
            )
        )


class TimeAtariiIQN(TimeAtariQ):
    def __init__(self) -> None:
        self.random_seed = RANDOM_SEED
        print(f"random seed {self.random_seed}", end=" ")
        self.key = jax.random.PRNGKey(self.random_seed)
        self.n_heads = int(jax.random.randint(self.key, (), minval=5, maxval=20))
        print(f"{self.n_heads} heads")
        self.state_shape = (84, 84, 4)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.cumulative_gamma = jax.random.uniform(self.key)
        self.head_behaviorial_probability = jax.random.uniform(self.key, (self.n_heads,), minval=1, maxval=10)
        shared_network = True
        print("Shared network" if shared_network else "Independant network")
        super().__init__(
            q=AtariiIQN(
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
                32,
                32,
                32,
                shared_network,
            )
        )

    def time_inference(self) -> None:
        state_key = self.key
        apply_func = jax.jit(self.q.apply_n_quantiles)

        # Outside of the count: time to jit the __call__ function
        jax.block_until_ready(
            apply_func(
                self.q.params, jax.random.uniform(state_key, (self.batch_size,) + self.state_shape), self.q.network_key
            )
        )

        t_begin = time()

        for _ in range(self.n_runs):
            state_key, key = jax.random.split(state_key)
            jax.block_until_ready(
                apply_func(
                    self.q.params, jax.random.uniform(key, (self.batch_size,) + self.state_shape), self.q.network_key
                )
            )

        print("Time inference: ", (time() - t_begin) / self.n_runs)


class TimeAtariiREM(TimeAtariQ):
    def __init__(self) -> None:
        self.random_seed = RANDOM_SEED
        print(f"random seed {self.random_seed}", end=" ")
        self.key = jax.random.PRNGKey(self.random_seed)
        self.n_heads = int(jax.random.randint(self.key, (), minval=5, maxval=20))
        print(f"{self.n_heads} heads")
        self.state_shape = (84, 84, 4)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.cumulative_gamma = jax.random.uniform(self.key)
        self.head_behaviorial_probability = jax.random.uniform(self.key, (self.n_heads,), minval=1, maxval=10)
        shared_network = True
        print("Shared network" if shared_network else "Independant network")
        super().__init__(
            q=AtariiREM(
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
                shared_network,
            )
        )
