import unittest
import jax
import numpy as np

from idqn.environments.atari import AtariEnv


class TestAtariEnv(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        print(f"random seed {self.random_seed}")
        self.key = jax.random.PRNGKey(self.random_seed)
        self.env = AtariEnv("Breakout")

    def test_reset(self) -> None:
        self.env.reset()
        for i in range(70):
            self.env.step(i % 4)
        state = self.env.state

        self.env.reset()
        for i in range(70):
            self.env.step(i % 4)
        state_bis = self.env.state

        self.assertNotEqual(np.linalg.norm(np.array(state) - np.array(state_bis)), 0)

    def test_step_frame_stacking(self) -> None:
        action_key = self.key
        self.env.reset()
        absorbing = False

        while not absorbing:
            action_key, key = jax.random.split(action_key)
            action = jax.random.randint(key, shape=(), minval=0, maxval=self.env.n_actions)
            _, absorbing, _ = self.env.step(action)

            self.assertEqual(self.env.state.shape[-1], self.env.n_stacked_frames)

        self.assertEqual(self.env.env.unwrapped.ale.lives(), 0)
