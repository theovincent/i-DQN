import os
from typing import Tuple
from functools import partial
from gymnasium.wrappers.monitoring import video_recorder
import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
from collections import deque
import cv2

from idqn.networks.base_q import BaseMultiHeadQ


class AtariEnv:
    def __init__(self, env_key: jax.random.PRNGKeyArray, name: str, gamma: float) -> None:
        self.reset_key = env_key
        self.name = name
        self.gamma = gamma
        self.state_height, self.state_width = (84, 84)
        self.n_stacked_frames = 4
        self.n_skipped_frames = 4
        self.max_no_op_actions = 30

        self.env = gym.make(
            f"ALE/{self.name}",
            full_action_space=False,
            frameskip=1,
            repeat_action_probability=0.25,
            render_mode="rgb_array",
        )
        self.n_actions = self.env.env.action_space.n
        self.original_state_height, self.original_state_width, _ = self.env.env.observation_space._shape

        _, info = self.env.reset()
        self.n_lives = info["lives"]

    def reset(self) -> np.ndarray:
        self.reset_key, key = jax.random.split(self.reset_key)
        frame, _ = self.env.reset(seed=int(key[0]))

        for _ in np.arange(jax.random.randint(key, (), 0, self.max_no_op_actions)):
            frame = self.env.step(0)[0]

        self.stacked_frames = deque(
            np.repeat(self.preprocess_frame(frame)[None, ...], self.n_stacked_frames, axis=0),
            maxlen=self.n_stacked_frames,
        )
        self.state = np.array(self.stacked_frames)

        self.n_steps = 0

        return self.state

    def step(self, action: jnp.int8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        absorbing = False
        n_steps = 0
        skipped_frames = np.zeros((self.n_skipped_frames, self.original_state_height, self.original_state_width, 3))
        reward = 0

        while n_steps < self.n_skipped_frames and not absorbing:
            skipped_frames[n_steps], reward_, absorbing_, _, info = self.env.step(action)

            n_steps += 1
            absorbing = absorbing_ or (info["lives"] < self.n_lives)
            reward += reward_

        # if there is less than n_skipped_frames frames, the max pooling eliminates the zeros
        self.stacked_frames.append(self.preprocess_frame(np.max(skipped_frames, axis=0)))
        self.state = np.array(self.stacked_frames)

        self.n_steps += self.n_skipped_frames

        return self.state, np.array(np.sign(reward), dtype=np.int8), np.array(absorbing, dtype=np.bool_), _

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        grayscaled_frame = cv2.cvtColor(np.array(frame, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        return cv2.resize(grayscaled_frame, (self.state_width, self.state_height), interpolation=cv2.INTER_LINEAR)

    @partial(jax.jit, static_argnames=("self", "q"))
    def best_action_multi_head(
        self, q: BaseMultiHeadQ, idx_head: int, q_params: hk.Params, state: np.ndarray
    ) -> jnp.int8:
        return q(q_params, jnp.array(state, dtype=jnp.float32))[0, idx_head].argmax()

    @partial(jax.jit, static_argnames="self")
    def random_action(self, key: jax.random.PRNGKeyArray) -> jnp.int8:
        return jax.random.choice(key, jnp.arange(self.n_actions))

    @partial(jax.jit, static_argnames=("self", "n_heads"))
    def random_head(
        self, key: jax.random.PRNGKeyArray, n_heads: int, head_behaviorial_probability: jnp.ndarray
    ) -> jnp.int8:
        return jax.random.choice(key, jnp.arange(n_heads), p=head_behaviorial_probability)

    def evaluate_(self, q: BaseMultiHeadQ, idx_head: int, q_params: hk.Params, horizon: int, video_path: str) -> float:
        if video_path is not None:
            video = video_recorder.VideoRecorder(
                self.env, path=f"experiments/atari/figures/{video_path}.mp4", disable_logger=True
            )
        cumulative_reward = 0
        discount = 1
        absorbing = False
        self.reset()

        while not absorbing and self.n_steps < horizon:
            if video_path is not None:
                self.env.render()
                video.capture_frame()

            action = self.best_action_multi_head(q, idx_head, q_params, self.state)
            _, reward, absorbing, _ = self.step(action)

            cumulative_reward += discount * reward
            discount *= self.gamma

        if video_path is not None:
            video.close()
            os.remove(f"experiments/atari/figures/{video_path}.meta.json")

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
