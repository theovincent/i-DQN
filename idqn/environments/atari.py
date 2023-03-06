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
from idqn.utils.pickle import save_pickled_data, load_pickled_data


class AtariEnv:
    def __init__(self, env_key: jax.random.PRNGKeyArray, name: str, gamma: float, terminal_on_life_loss: bool) -> None:
        self.reset_key = env_key
        self.name = name
        self.gamma = gamma
        self.terminal_on_life_loss = terminal_on_life_loss
        self.state_height, self.state_width = (84, 84)
        self.n_stacked_frames = 4
        self.n_skipped_frames = 4
        self.n_pooled_frames = 2

        self.env = gym.make(
            f"ALE/{self.name}",
            full_action_space=False,
            frameskip=1,
            repeat_action_probability=0.25,
            obs_type="grayscale",
            render_mode="rgb_array",
        )
        self.n_actions = self.env.env.action_space.n
        self.original_state_height, self.original_state_width = self.env.env.observation_space._shape

        self.start_with_fire = self.env.unwrapped.get_action_meanings()[1] == "FIRE"
        _, info = self.env.reset()
        self.n_lives = info["lives"]

    def reset(self) -> np.ndarray:
        self.reset_key, key = jax.random.split(self.reset_key)
        frame, _ = self.env.reset(seed=int(key[0]))

        self.stacked_frames = deque(
            np.repeat(self.preprocess_frame(frame)[None, ...], self.n_stacked_frames, axis=0),
            maxlen=self.n_stacked_frames,
        )
        self.state = np.array(self.stacked_frames)

        self.n_steps = 0

        return self.state

    @partial(jax.jit, static_argnames="self")
    def is_absorbing(self, absorbing_, info, n_lives):
        if self.terminal_on_life_loss:
            return absorbing_ or (info["lives"] < n_lives), info["lives"]
        else:
            return absorbing_, info["lives"]

    def step(self, action: jnp.int8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        absorbing = False
        n_frames = 0
        pooled_frames = np.zeros((self.n_pooled_frames, self.original_state_height, self.original_state_width))
        reward = 0

        while n_frames < self.n_skipped_frames and not absorbing:
            pooled_frames[n_frames % self.n_pooled_frames], reward_, absorbing_, _, info = self.env.step(action)

            n_frames += 1
            absorbing, self.n_lives = self.is_absorbing(absorbing_, info, self.n_lives)
            reward += reward_

        # if there is less than n_skipped_frames frames, the max pooling eliminates the zeros
        self.stacked_frames.append(self.preprocess_frame(np.max(pooled_frames, axis=0)))
        self.state = np.array(self.stacked_frames)

        self.n_steps += 1

        return self.state, np.array(np.clip(reward, -1, 1), dtype=np.int8), np.array(absorbing, dtype=np.bool_), _

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(
            np.array(frame, dtype=np.uint8), (self.state_width, self.state_height), interpolation=cv2.INTER_LINEAR
        )

    def get_ale_state(self):
        return self.env.unwrapped.clone_state(include_rng=True)

    def restore_ale_state(self, env_state) -> None:
        self.env.unwrapped.restore_state(env_state)

    def save(self, path: str) -> None:
        save_pickled_data(path + "_ale_state", self.get_ale_state())
        save_pickled_data(path + "_frame_state", self.stacked_frames)
        save_pickled_data(path + "_n_steps", self.n_steps)

    def load(self, path: str) -> None:
        self.restore_ale_state(load_pickled_data(path + "_ale_state"))
        self.stacked_frames = load_pickled_data(path + "_frame_state")
        self.state = np.array(self.stacked_frames)
        self.n_steps = load_pickled_data(path + "_n_steps")

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

    def evaluate_(
        self,
        q: BaseMultiHeadQ,
        idx_head: int,
        q_params: hk.Params,
        horizon: int,
        eps_eval: float,
        exploration_key: jax.random.PRNGKey,
        video_path: str,
    ) -> float:
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

            exploration_key, key = jax.random.split(exploration_key)
            if jax.random.uniform(key) < eps_eval:
                action = self.random_action(key)
            else:
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
        eps_eval: float,
        exploration_key: jax.random.PRNGKey,
        video_path: str,
    ) -> float:
        rewards = np.zeros(n_simulations)

        rewards[0] = self.evaluate_(q, idx_head, q_params, horizon, eps_eval, exploration_key, video_path)
        for idx_simulation in range(1, n_simulations):
            rewards[idx_simulation] = self.evaluate_(q, idx_head, q_params, horizon, eps_eval, exploration_key, None)

        return rewards.mean()
