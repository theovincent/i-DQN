"""
The environment is inspired from https://github.com/google/dopamine/blob/master/dopamine/discrete_domains/atari_lib.py
"""
import os
from typing import Tuple, Dict
from gymnasium.wrappers.monitoring import video_recorder
import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
from collections import deque
import cv2
from tqdm import tqdm

from idqn.networks.base_q import BaseQ
from idqn.sample_collection.replay_buffer import ReplayBuffer
from idqn.sample_collection.exploration import EpsilonGreedySchedule
from idqn.utils.pickle import save_pickled_data, load_pickled_data


class LazyFrames:
    """
    Ensures common frames are only stored once to optimize memory use.
    Inspired by https://github.com/Farama-Foundation/Gymnasium.
    """

    def __init__(self, frames: list):
        self._frames = frames

    def __array__(self):
        return np.array(self._frames)


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
    def state(self) -> np.ndarray:
        return np.array(self.stacked_frames)

    def reset(self) -> np.ndarray:
        self.env.reset()

        self.n_steps = 0

        self.env.ale.getScreenGrayscale(self.screen_buffer[0])
        self.screen_buffer[1].fill(0)

        self.stacked_frames = deque(
            np.repeat(self.resize()[None, ...], self.n_stacked_frames, axis=0),
            maxlen=self.n_stacked_frames,
        )

        return LazyFrames(list(self.stacked_frames))

    def step(self, action: jnp.int8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        reward = 0

        for idx_frame in range(self.n_skipped_frames):
            _, reward_, absorbing, _, _ = self.env.step(action)

            reward += reward_

            if idx_frame >= self.n_skipped_frames - 2:
                t = idx_frame - (self.n_skipped_frames - 2)
                self.env.ale.getScreenGrayscale(self.screen_buffer[t])

            if absorbing:
                break

        self.stacked_frames.append(self.pool_and_resize())

        self.n_steps += 1

        return LazyFrames(list(self.stacked_frames)), reward, absorbing, _

    def pool_and_resize(self) -> np.ndarray:
        np.maximum(self.screen_buffer[0], self.screen_buffer[1], out=self.screen_buffer[0])

        return self.resize()

    def resize(self):
        return np.asarray(
            cv2.resize(self.screen_buffer[0], (self.state_width, self.state_height), interpolation=cv2.INTER_AREA),
            dtype=np.uint8,
        )

    def get_ale_state(self):
        return self.env.ale.cloneSystemState()

    def restore_ale_state(self, env_state) -> None:
        self.env.ale.restoreSystemState(env_state)

    def save(self, path: str) -> None:
        save_pickled_data(path + "_ale_state", self.get_ale_state())
        save_pickled_data(path + "_frame_state", self.stacked_frames)
        save_pickled_data(path + "_n_steps", self.n_steps)

    def load(self, path: str) -> None:
        self.reset()
        self.restore_ale_state(load_pickled_data(path + "_ale_state"))
        self.stacked_frames = load_pickled_data(path + "_frame_state")
        self.n_steps = load_pickled_data(path + "_n_steps")

    def collect_random_samples(
        self, sample_key: jax.random.PRNGKeyArray, replay_buffer: ReplayBuffer, n_samples: int, horizon: int
    ) -> None:
        self.reset()

        for _ in tqdm(range(n_samples)):
            state = self.state

            sample_key, key = jax.random.split(sample_key)
            action = jax.random.choice(key, jnp.arange(self.n_actions))
            next_state, reward, absorbing, _ = self.step(action)

            replay_buffer.add(state, action, reward, next_state, absorbing)

            if absorbing or self.n_steps >= horizon:
                self.reset()

    def collect_one_sample(
        self,
        q: BaseQ,
        q_params: Dict,
        horizon: int,
        replay_buffer: ReplayBuffer,
        exploration_schedule: EpsilonGreedySchedule,
    ) -> bool:
        state = self.state

        if exploration_schedule.explore():
            action = q.random_action(exploration_schedule.key)
        else:
            action = q.best_action(exploration_schedule.key, q_params, self.state)

        next_state, reward, absorbing, _ = self.step(action)

        replay_buffer.add(state, action, reward, next_state, absorbing)

        if absorbing or self.n_steps >= horizon:
            self.reset()

        return reward, absorbing

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
        absorbing = False
        self.reset()

        while not absorbing and self.n_steps < horizon:
            self.env.render()
            video.capture_frame()

            exploration_key, key = jax.random.split(exploration_key)
            if jax.random.uniform(key) < eps_eval:
                action = q.random_action(key)
            else:
                action = q.best_action(key, q_params, self.state)

            _, reward, absorbing, _ = self.step(action)

            sun_reward += reward

        video.close()
        os.remove(f"experiments/atari/figures/{video_path}.meta.json")

        return sun_reward
