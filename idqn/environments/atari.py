"""
The environment is inspired from https://github.com/google/dopamine/blob/master/dopamine/discrete_domains/atari_lib.py
"""
import os
from typing import Tuple, Dict
from gym.wrappers.monitoring import video_recorder
import gym as gym
import numpy as np
import jax
import jax.numpy as jnp
import cv2
from tqdm import tqdm

from idqn.networks.base import BaseQ
from idqn.sample_collection.replay_buffer import ReplayBuffer
from idqn.sample_collection.exploration import EpsilonGreedySchedule


class AtariEnv:
    def __init__(
        self,
        name: str,
    ) -> None:
        self.name = name
        self.state_height, self.state_width = (84, 84)
        self.n_stacked_frames = 4
        self.n_skipped_frames = 4

        self.env = gym.make(
            f"ALE/{self.name}-v5",
            full_action_space=False,
            frameskip=1,
            repeat_action_probability=0.25,
            render_mode="rgb_array",
        ).env

        self.n_actions = self.env.action_space.n
        self.original_state_height, self.original_state_width, _ = self.env.observation_space._shape
        self.screen_buffer = [
            np.empty((self.original_state_height, self.original_state_width), dtype=np.uint8),
            np.empty((self.original_state_height, self.original_state_width), dtype=np.uint8),
        ]

    @property
    def observation(self) -> np.ndarray:
        return np.copy(self.state_[:, :, -1])

    @property
    def state(self) -> np.ndarray:
        return jnp.array(self.state_, dtype=jnp.float32)

    def reset(self) -> None:
        self.env.reset()

        self.n_steps = 0

        self.env.ale.getScreenGrayscale(self.screen_buffer[0])
        self.screen_buffer[1].fill(0)

        self.state_ = np.zeros((self.state_height, self.state_width, self.n_stacked_frames), dtype=np.uint8)
        self.state_[:, :, -1] = self.resize()

    def step(self, action: jnp.int8) -> Tuple[float, bool, Dict]:
        reward = 0

        for idx_frame in range(self.n_skipped_frames):
            _, reward_, terminal, _ = self.env.step(action)

            reward += reward_

            if idx_frame >= self.n_skipped_frames - 2:
                t = idx_frame - (self.n_skipped_frames - 2)
                self.env.ale.getScreenGrayscale(self.screen_buffer[t])

            if terminal:
                break

        self.state_ = np.roll(self.state_, -1, axis=-1)
        self.state_[:, :, -1] = self.pool_and_resize()

        self.n_steps += 1

        return reward, terminal, _

    def pool_and_resize(self) -> np.ndarray:
        np.maximum(self.screen_buffer[0], self.screen_buffer[1], out=self.screen_buffer[0])

        return self.resize()

    def resize(self):
        return np.asarray(
            cv2.resize(self.screen_buffer[0], (self.state_width, self.state_height), interpolation=cv2.INTER_AREA),
            dtype=np.uint8,
        )

    def collect_random_samples(
        self, sample_key: jax.random.PRNGKeyArray, replay_buffer: ReplayBuffer, n_samples: int, horizon: int
    ) -> None:
        self.reset()

        for _ in tqdm(range(n_samples)):
            observation = self.observation

            sample_key, key = jax.random.split(sample_key)
            action = jax.random.choice(key, jnp.arange(self.n_actions))
            reward, terminal, _ = self.step(action)

            episode_end = terminal or self.n_steps >= horizon
            replay_buffer.add(observation, action, reward, terminal, episode_end=episode_end)

            if episode_end:
                self.reset()

    def collect_one_sample(
        self,
        q: BaseQ,
        q_params: Dict,
        horizon: int,
        replay_buffer: ReplayBuffer,
        exploration_schedule: EpsilonGreedySchedule,
    ) -> Tuple[float, bool]:
        observation = self.observation

        if exploration_schedule.explore():
            action = q.random_action(exploration_schedule.key)
        else:
            action = q.best_action(q_params, self.state, key=exploration_schedule.key)

        reward, terminal, _ = self.step(action)

        episode_end = terminal or self.n_steps >= horizon
        replay_buffer.add(observation, action, reward, terminal, episode_end=episode_end)

        if episode_end:
            self.reset()

        return reward, episode_end

    def evaluate_one_simulation(
        self,
        q: BaseQ,
        q_params: Dict,
        horizon: int,
        eps_eval: float,
        exploration_key: jax.random.PRNGKey,
        video_path: str,
    ) -> float:
        video = video_recorder.VideoRecorder(
            self.env, path=f"experiments/atari/figures/{video_path}.mp4", disable_logger=True
        )
        sun_reward = 0
        terminal = False
        self.reset()

        while not terminal and self.n_steps < horizon:
            self.env.render()
            video.capture_frame()

            exploration_key, key = jax.random.split(exploration_key)
            if jax.random.uniform(key) < eps_eval:
                action = q.random_action(key)
            else:
                action = q.best_action(q_params, self.state, key=key)

            reward, terminal, _ = self.step(action)

            sun_reward += reward

        video.close()
        os.remove(f"experiments/atari/figures/{video_path}.meta.json")

        return sun_reward, terminal
