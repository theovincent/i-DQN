from time import time
import jax
import numpy as np

from idqn.environments.atari import AtariEnv


def run_cli():
    time_atari_env = TimeAtariEnv()

    time_atari_env.time_reset()
    time_atari_env.time_step()


class TimeAtariEnv:
    def __init__(self) -> None:
        self.random_seed = np.random.randint(100)
        print(f"random seed {self.random_seed}")
        self.n_runs = 1000
        self.key = jax.random.PRNGKey(self.random_seed)
        self.name = "Breakout"

    def time_reset(self) -> None:
        env = AtariEnv(self.name)

        t_begin = time()

        for _ in range(self.n_runs):
            env.reset()

        print("Time reset: ", (time() - t_begin) / self.n_runs)

    def time_step(self) -> None:
        env = AtariEnv(self.name)
        env.reset()
        action_key = self.key

        t_begin = time()

        for _ in range(self.n_runs):
            action_key, key = jax.random.split(action_key)
            action = jax.random.randint(key, shape=(), minval=0, maxval=env.n_actions)
            _, _, _, _ = env.step(action)

        print("Time step: ", (time() - t_begin) / self.n_runs)
