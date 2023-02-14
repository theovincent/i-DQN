from typing import Tuple
from functools import partial
from gymnasium.wrappers.monitoring import video_recorder
import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

from idqn.networks.base_q import BaseMultiHeadQ


class LunarLanderEnv:
    def __init__(self, env_key: jax.random.PRNGKeyArray, gamma: float) -> None:
        self.n_actions = 4
        self.state_shape = 8
        self.gamma = gamma
        self.reset_key = env_key

        self.env = gym.make("LunarLander-v2", render_mode="rgb_array")

    def reset(self) -> jnp.ndarray:
        self.reset_key, key = jax.random.split(self.reset_key)
        self.state, _ = self.env.reset(seed=int(key[0]))
        self.n_steps = 0

        return jnp.array(self.state)

    def step(self, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        self.state, reward, absorbing, _, info = self.env.step(int(action[0]))
        self.n_steps += 1

        return jnp.array(self.state), reward, absorbing, info

    @partial(jax.jit, static_argnames=("self", "q"))
    def jitted_best_action_multi_head(
        self, q: BaseMultiHeadQ, idx_head: int, q_params: hk.Params, state: jnp.ndarray
    ) -> jnp.ndarray:
        return q(q_params, state)[0, idx_head].argmax()

    def evaluate_(self, q: BaseMultiHeadQ, idx_head: int, q_params: hk.Params, horizon: int, video_path: str) -> float:
        if video_path is not None:
            video = video_recorder.VideoRecorder(
                self.env, path=f"experiments/lunar_lander/figures/{video_path}.mp4", disable_logger=True
            )
        cumulative_reward = 0
        discount = 1
        absorbing = False
        self.reset()

        while not absorbing and self.n_steps < horizon:
            if video_path is not None:
                self.env.render()
                video.capture_frame()

            action = self.jitted_best_action_multi_head(q, idx_head, q_params, self.state)
            _, reward, absorbing, _ = self.step(action)

            cumulative_reward += discount * reward
            discount *= self.gamma

        if video_path is not None:
            self.env.close()
            video.close()

        return cumulative_reward

    def evaluate(
        self,
        q: BaseMultiHeadQ,
        idx_head: int,
        q_params: hk.Params,
        horizon: int,
        n_simulations: int,
        video_path: str,
    ) -> float:
        rewards = np.zeros(n_simulations)

        rewards[0] = self.evaluate_(q, idx_head, q_params, horizon, video_path)
        for idx_simulation in range(1, n_simulations):
            rewards[idx_simulation] = self.evaluate_(q, idx_head, q_params, horizon, None)

        return rewards.mean()
