import os
import unittest
import jax
import numpy as np
import gymnasium as gym

from idqn.environments.atari import AtariEnv


class TestAtariEnv(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = 272  # np.random.randint(1000)
        self.key = jax.random.PRNGKey(self.random_seed)
        self.name = "Breakout"
        self.gamma = jax.random.uniform(self.key)
        self.start_with_fire = jax.random.randint(self.key, (), minval=0, maxval=2).astype(bool)
        key, _ = jax.random.split(self.key)
        self.terminal_on_life_loss = jax.random.randint(key, (), minval=0, maxval=2).astype(bool)

    def test_reset(self) -> None:
        # Force hard reset
        env = AtariEnv(
            self.key, self.name, self.gamma, start_with_fire=True, terminal_on_life_loss=self.terminal_on_life_loss
        )

        env.reset()
        for i in range(7):
            state = env.step(i % 4)[0]

        env.reset(truncation=True)
        for i in range(7):
            state_bis = env.step(i % 4)[0]

        self.assertNotEqual(np.linalg.norm(state - state_bis), 0, f"random seed {self.random_seed}")

        # Force hard reset
        env = AtariEnv(self.key, self.name, self.gamma, start_with_fire=True, terminal_on_life_loss=False)

        env.reset()
        for i in range(7):
            state = env.step(i % 4)[0]

        env.reset()
        for i in range(7):
            state_bis = env.step(i % 4)[0]

        self.assertNotEqual(np.linalg.norm(state - state_bis), 0, f"random seed {self.random_seed}")

        # Force soft reset
        env = AtariEnv(self.key, self.name, self.gamma, start_with_fire=False, terminal_on_life_loss=True)
        env.n_pooled_frames = 1

        env.reset()
        for i in range(7):
            state = env.step(i % 4)[0]

        state_bis = env.reset()

        self.assertEqual(np.linalg.norm(state[-1] - state_bis[-1]), 0, f"random seed {self.random_seed}")

    def test_step_deterministic_stochasticity(self) -> None:
        env = AtariEnv(self.key, self.name, self.gamma, self.start_with_fire, self.terminal_on_life_loss)
        state = env.reset()
        successive_states = np.zeros(((10,) + state.shape))
        successive_actions = jax.random.randint(self.key, (10,), 0, env.n_actions)

        for step in range(10):
            successive_states[step] = env.step(successive_actions[step])[0]

        env_bis = AtariEnv(self.key, self.name, self.gamma, self.start_with_fire, self.terminal_on_life_loss)
        state_bis = env_bis.reset()
        successive_states_bis = np.zeros(((10,) + state_bis.shape))

        for step in range(10):
            successive_states_bis[step] = env_bis.step(successive_actions[step])[0]

        self.assertAlmostEqual(
            np.linalg.norm(successive_states - successive_states_bis), 0, f"random seed {self.random_seed}"
        )

    def test_step_frame_stacking(self) -> None:
        env = AtariEnv(self.key, self.name, self.gamma, self.start_with_fire, self.terminal_on_life_loss)
        action_key = self.key
        env.reset()
        absorbing = False

        while not absorbing:
            action_key, key = jax.random.split(action_key)
            action = jax.random.randint(key, shape=(), minval=0, maxval=env.n_actions)
            state, _, absorbing, _ = env.step(action)

            self.assertEqual(state.shape[0], env.n_stacked_frames, f"random seed {self.random_seed}")

        self.assertEqual(
            env.env.unwrapped.ale.lives(), 4 if self.terminal_on_life_loss else 0, f"random seed {self.random_seed}"
        )

    def test_store_load(self) -> None:
        # Need to remove stochastic actions
        env_to_store = AtariEnv(
            self.key,
            self.name,
            self.gamma,
            start_with_fire=self.start_with_fire,
            terminal_on_life_loss=self.terminal_on_life_loss,
        )
        env_to_store.env = gym.make(
            f"ALE/{self.name}-v5",
            full_action_space=False,
            frameskip=1,
            repeat_action_probability=0,
            obs_type="grayscale",
            render_mode="rgb_array",
        )
        action_key = self.key
        env_to_store.reset()

        for _ in range(10):
            action_key, key = jax.random.split(action_key)
            env_to_store.step(jax.random.randint(key, shape=(), minval=0, maxval=env_to_store.n_actions))

        env_to_store.save("test/test_store_load")

        env_to_load = AtariEnv(
            self.key,
            self.name,
            self.gamma,
            start_with_fire=self.start_with_fire,
            terminal_on_life_loss=self.terminal_on_life_loss,
        )
        env_to_load.env = gym.make(
            f"ALE/{self.name}-v5",
            full_action_space=False,
            frameskip=1,
            repeat_action_probability=0,
            obs_type="grayscale",
            render_mode="rgb_array",
        )
        env_to_load.load("test/test_store_load")

        self.assertEqual(np.linalg.norm(env_to_store.state - env_to_load.state), 0, f"random seed {self.random_seed}")
        self.assertEqual(env_to_store.n_steps, env_to_load.n_steps, f"random seed {self.random_seed}")

        env_to_store.step(0)
        env_to_load.step(0)

        self.assertEqual(np.linalg.norm(env_to_store.state - env_to_load.state), 0, f"random seed {self.random_seed}")

        os.remove("test/test_store_load_ale_state")
        os.remove("test/test_store_load_frame_state")
        os.remove("test/test_store_load_n_steps")
