import os
import unittest
import jax
import numpy as np
import gymnasium as gym

from idqn.environments.atari import AtariEnv


class TestAtariEnv(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        print(f"random seed {self.random_seed}")
        self.key = jax.random.PRNGKey(self.random_seed)
        self.name = "Breakout"

    def test_reset(self) -> None:
        env = AtariEnv(self.name)

        env.reset()
        for i in range(70):
            state = env.step(i % 4)[0]

        env.reset()
        for i in range(70):
            state_bis = env.step(i % 4)[0]

        self.assertNotEqual(np.linalg.norm(state - state_bis), 0)

    def test_step_frame_stacking(self) -> None:
        env = AtariEnv(self.name)
        action_key = self.key
        env.reset()
        absorbing = False

        while not absorbing:
            action_key, key = jax.random.split(action_key)
            action = jax.random.randint(key, shape=(), minval=0, maxval=env.n_actions)
            state, _, absorbing, _ = env.step(action)

            self.assertEqual(state.shape[0], env.n_stacked_frames)

        self.assertEqual(env.env.unwrapped.ale.lives(), 0)

    def test_store_load(self) -> None:
        # Need to remove stochastic actions
        env_to_store = AtariEnv(self.name)
        env_to_store.env = gym.make(
            f"ALE/{self.name}-v5",
            full_action_space=False,
            frameskip=1,
            repeat_action_probability=0,
            render_mode="rgb_array",
        )
        action_key = self.key
        env_to_store.reset()

        for _ in range(10):
            action_key, key = jax.random.split(action_key)
            env_to_store.step(jax.random.randint(key, shape=(), minval=0, maxval=env_to_store.n_actions))

        env_to_store.save("tests/test_store_load")

        env_to_load = AtariEnv(self.name)
        env_to_load.env = gym.make(
            f"ALE/{self.name}-v5",
            full_action_space=False,
            frameskip=1,
            repeat_action_probability=0,
            render_mode="rgb_array",
        )
        env_to_load.load("tests/test_store_load")

        self.assertEqual(np.linalg.norm(env_to_store.state - env_to_load.state), 0)
        self.assertEqual(env_to_store.n_steps, env_to_load.n_steps)

        env_to_store.step(0)
        env_to_load.step(0)

        self.assertEqual(np.linalg.norm(env_to_store.state - env_to_load.state), 0)

        os.remove("tests/test_store_load_ale_state")
        os.remove("tests/test_store_load_frame_state")
        os.remove("tests/test_store_load_n_steps")
