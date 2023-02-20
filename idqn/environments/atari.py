import os
from typing import Tuple
from functools import partial
from gymnasium.wrappers.monitoring import video_recorder
import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
from PIL import Image
from collections import deque

from idqn.networks.base_q import BaseMultiHeadQ


class AtariEnv:
    def __init__(self, env_key: jax.random.PRNGKeyArray, name: str, gamma: float) -> None:
        self.reset_key = env_key
        self.name = name
        self.gamma = gamma
        self.state_height, self.state_width = (84, 84)
        self.n_stacked_frames = 4
        self.n_skipped_frames = 4
        self.max_abs_reward = 1

        self.env = gym.make(f"ALE/{self.name}", full_action_space=False, obs_type="grayscale", frameskip=1)
        self.n_actions = self.self.env.env.observation_space._shape.env.env.action_space.n
        self.original_state_height, self.original_state_width = self.env.env.env.observation_space._shape

        _, info = self.env.reset()
        self.n_lives = info["lives"]

    def reset(self) -> jnp.ndarray:
        self.reset_key, key = jax.random.split(self.reset_key)
        frame, _ = self.env.reset(seed=int(key[0]))

        state = Image.fromarray(frame).resize((self.state_width, self.state_height), Image.Resampling.BILINEAR)
        self.state = jnp.array(np.repeat(state[None, ...], self.n_stacked_frames, axis=0), dtype=jnp.float32)

        # deque(maxlen=self._num_frames)

        self.n_steps = 0

        return self.state

    def step(self, action: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        absorbing = False
        n_steps = 0
        frames = np.zeros((self.n_stacked_frames, self.original_state_height, self.original_state_width))
        reward = 0

        while n_steps < self.n_stacked_frames and not absorbing:
            frames[n_steps], reward_, absorbing_, _, info = self.env.step(int(action))

            n_steps += 1
            absorbing = absorbing_ or (info["lives"] < self.n_lives)
            reward += reward_

        for n_steps_ in np.arange(n_steps + 1, self.n_stacked_frames):
            frames[n_steps_] = frames[n_steps]

        pooled_frames = np.max(frames, axis=0)
        self.state = jnp.array(
            Image.fromarray(pooled_frames).resize((self.state_width, self.state_height), Image.Resampling.BILINEAR)
            / 255.0,
            dtype=jnp.float32,
        )

        self.n_steps += self.n_stacked_frames

        return (
            self.state,
            jnp.array(np.clip(reward, -self.max_abs_reward, self.max_abs_reward), dtype=jnp.float32),
            jnp.array(absorbing, dtype=bool),
            _,
        )

    @partial(jax.jit, static_argnames=("self", "q"))
    def jitted_best_action_multi_head(
        self, q: BaseMultiHeadQ, idx_head: int, q_params: hk.Params, state: jnp.ndarray
    ) -> jnp.ndarray:
        return q(q_params, state)[0, idx_head].argmax()

    def evaluate_(self, q: BaseMultiHeadQ, idx_head: int, q_params: hk.Params, horizon: int, video_path: str) -> float:
        if video_path is not None:
            video = video_recorder.VideoRecorder(
                self.env, path=f"experiments/atari/figures/{self.name}/{video_path}.mp4", disable_logger=True
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
            video.close()
            os.remove(f"experiments/atari/figures/{self.name}/{video_path}.meta.json")

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
