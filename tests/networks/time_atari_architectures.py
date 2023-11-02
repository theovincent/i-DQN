from typing import Tuple, Callable
from time import time
import jax
import numpy as np

from idqn.networks.base import BaseQ
from idqn.networks.architectures.dqn import AtariDQN
from idqn.networks.architectures.iqn import AtariIQN
from idqn.networks.architectures.rem import AtariREM
from idqn.networks.architectures.idqn import AtariiDQN
from idqn.networks.architectures.iiqn import AtariiIQN
from idqn.networks.architectures.irem import AtariiREM
from tests.networks.utils import Generator

RANDOM_SEED = 845  # np.random.randint(1000)


def run_cli():
    print(f"random seed {RANDOM_SEED}")

    timers = [
        TimeAtariDQN(),
        TimeAtariIQN(),
        TimeAtariREM(),
        TimeAtariiDQN(),
        TimeAtariiIQN(),
        TimeAtariiREM(),
    ]

    print("\n\nTime apply")
    for timer in timers:
        timer.time_apply()

    print("\n\nTime compute target")
    for timer in timers:
        timer.time_compute_target()

    print("\n\nTime compute gradient")
    for timer in timers:
        timer.time_compute_gradient()

    print("\n\nTime best action")
    for timer in timers:
        timer.time_best_action()


class TimeAtariQ:
    def __init__(self, q: BaseQ, state_shape: Tuple) -> None:
        self.n_runs = 6000
        self.q = q
        self.key = q.network_key
        self.n_actions = q.n_actions
        self.generator = Generator(32, state_shape)

    def base_timer(self, func: Callable, args_builder: Callable, generator: Callable) -> None:
        key = self.key

        # Outside of the count: time to jit the function
        args = args_builder(self.q.params, self.q.params, generator(key), key)
        jax.block_until_ready(func(*args))

        t_begin = time()

        for _ in range(self.n_runs):
            key, key_ = jax.random.split(key)
            args = args_builder(self.q.params, self.q.params, generator(key_), key_)
            jax.block_until_ready(func(*args))

        print(f"{self.algorithm}: ", (time() - t_begin) / self.n_runs)

    def time_apply(self) -> None:
        apply_func = jax.jit(jax.vmap(self.q.apply, in_axes=(None, 0)))
        # apply_func only needs params and samples
        args_builder = lambda params, target_params, samples, key: (params, samples)

        self.base_timer(apply_func, args_builder, self.generator.generate_states)

    def time_compute_target(self) -> None:
        compute_target_func = jax.jit(jax.vmap(self.q.compute_target, in_axes=(None, 0)))
        # compute_target_func only needs target_params and samples
        args_builder = lambda params, target_params, samples, key: (target_params, samples)

        self.base_timer(compute_target_func, args_builder, self.generator.generate_samples)

    def time_compute_gradient(self) -> None:
        loss_and_grad_func = jax.jit(jax.value_and_grad(self.q.loss_on_batch))
        args_builder = lambda *args: args

        self.base_timer(loss_and_grad_func, args_builder, self.generator.generate_samples)

    def time_best_action(self) -> None:
        # best_action only needs params, samples and key
        args_builder = lambda params, target_params, samples, key: (params, samples, key)

        self.base_timer(self.q.best_action, args_builder, self.generator.generate_state)


class TimeAtariQuantileQ(TimeAtariQ):
    def time_apply(self) -> None:
        apply_func = jax.jit(jax.vmap(self.q.apply, in_axes=(None, 0, None, None)), static_argnames="n_quantiles")
        # apply_func only needs params, samples, key and self.q.n_quantiles
        args_builder = lambda params, target_params, samples, key: (params, samples, key, self.q.n_quantiles)

        self.base_timer(apply_func, args_builder, self.generator.generate_states)

    def time_compute_target(self) -> None:
        compute_target_func = jax.jit(jax.vmap(self.q.compute_target, in_axes=(None, 0, None)))
        # compute_target_func only needs target_params and samples
        args_builder = lambda params, target_params, samples, key: (target_params, samples, key)

        self.base_timer(compute_target_func, args_builder, self.generator.generate_samples)


class TimeAtariDQN(TimeAtariQ):
    def __init__(self) -> None:
        self.algorithm = " DQN"
        self.random_seed = RANDOM_SEED
        self.key = jax.random.PRNGKey(self.random_seed)
        self.state_shape = (84, 84, 4)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.cumulative_gamma = jax.random.uniform(self.key)
        super().__init__(
            AtariDQN(self.state_shape, self.n_actions, self.cumulative_gamma, self.key, None, None, None, None)
        )


class TimeAtariIQN(TimeAtariQuantileQ):
    def __init__(self) -> None:
        self.algorithm = " IQN"
        self.random_seed = RANDOM_SEED
        self.key = jax.random.PRNGKey(self.random_seed)
        self.state_shape = (84, 84, 4)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.cumulative_gamma = jax.random.uniform(self.key)
        super().__init__(
            AtariIQN(self.state_shape, self.n_actions, self.cumulative_gamma, self.key, None, None, None, None)
        )


class TimeAtariREM(TimeAtariQ):
    def __init__(self) -> None:
        self.algorithm = " REM"
        self.random_seed = RANDOM_SEED
        self.key = jax.random.PRNGKey(self.random_seed)
        self.state_shape = (84, 84, 4)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.cumulative_gamma = jax.random.uniform(self.key)
        super().__init__(
            AtariREM(self.state_shape, self.n_actions, self.cumulative_gamma, self.key, None, None, None, None)
        )


class TimeAtariiDQN(TimeAtariQ):
    def __init__(self) -> None:
        self.algorithm = "iDQN"
        self.random_seed = RANDOM_SEED
        self.key = jax.random.PRNGKey(self.random_seed)
        self.n_heads = int(jax.random.randint(self.key, (), minval=5, maxval=20))
        print(f"iDQN with {self.n_heads} heads", end=" ")
        self.state_shape = (84, 84, 4)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.cumulative_gamma = jax.random.uniform(self.key)
        self.head_behaviorial_probability = jax.random.uniform(self.key, (self.n_heads,), minval=1, maxval=10)
        shared_network = True
        print("and shared networks" if shared_network else "and independant networks")
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


class TimeAtariiIQN(TimeAtariQuantileQ):
    def __init__(self) -> None:
        self.algorithm = "iIQN"
        self.random_seed = RANDOM_SEED
        self.key = jax.random.PRNGKey(self.random_seed)
        self.n_heads = int(jax.random.randint(self.key, (), minval=5, maxval=20))
        print(f"iIQN with {self.n_heads} heads", end=" ")
        self.state_shape = (84, 84, 4)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.cumulative_gamma = jax.random.uniform(self.key)
        self.head_behaviorial_probability = jax.random.uniform(self.key, (self.n_heads,), minval=1, maxval=10)
        shared_network = True
        print("and shared networks" if shared_network else "and independant networks")
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


class TimeAtariiREM(TimeAtariQ):
    def __init__(self) -> None:
        self.algorithm = "iREM"
        self.random_seed = RANDOM_SEED
        self.key = jax.random.PRNGKey(self.random_seed)
        self.n_heads = int(jax.random.randint(self.key, (), minval=5, maxval=20))
        print(f"iREM with {self.n_heads} heads", end=" ")
        self.state_shape = (84, 84, 4)
        self.n_actions = int(jax.random.randint(self.key, (), minval=1, maxval=10))
        self.cumulative_gamma = jax.random.uniform(self.key)
        self.head_behaviorial_probability = jax.random.uniform(self.key, (self.n_heads,), minval=1, maxval=10)
        shared_network = True
        print("and shared networks" if shared_network else "and independant networks")
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
