"""
The environment is inspired from https://github.com/google/dopamine/blob/master/dopamine/discrete_domains/atari_lib.py
"""
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
from tqdm import tqdm

from idqn.networks.base_q import BaseQ
from idqn.sample_collection.replay_buffer import ReplayBuffer
from idqn.sample_collection.exploration import EpsilonGreedySchedule
from idqn.utils.pickle import save_pickled_data, load_pickled_data


class AtariEnv:
    def __init__(
        self,
        env_key: jax.random.PRNGKeyArray,
        name: str,
        gamma: float,
        start_with_fire: bool,
        terminal_on_life_loss: bool,
    ) -> None:
        self.reset_key = env_key
        self.name = name
        self.gamma = gamma
        self.start_with_fire = start_with_fire
        self.terminal_on_life_loss = terminal_on_life_loss
        self.state_height, self.state_width = (84, 84)
        self.n_stacked_frames = 4
        self.n_skipped_frames = 4

        self.env = gym.make(
            f"ALE/{self.name}-v5",
            full_action_space=False,
            frameskip=1,
            repeat_action_probability=0.25,
            obs_type="grayscale",
            render_mode="rgb_array",
        ).env
        self.has_reset = False

        self.n_actions = self.env.action_space.n
        self.original_state_height, self.original_state_width = self.env.observation_space._shape
        self.screen_buffer = [
            np.empty((self.original_state_height, self.original_state_width), dtype=np.uint8),
            np.empty((self.original_state_height, self.original_state_width), dtype=np.uint8),
        ]

    def reset(self, truncation: bool = False) -> np.ndarray:
        # Avoid hard reset only when:
        #    - the environment has been hard reset AND
        #    - a terminal state is set at each life loss AND
        #    - the game is not over AND
        #    - no truncation wants to be done.
        if self.has_reset and self.terminal_on_life_loss and not self.env.ale.game_over() and not truncation:
            pass
        else:
            self.reset_key, key = jax.random.split(self.reset_key)
            _, info = self.env.reset(seed=int(key[0]))
            self.has_reset = True

            self.n_lives = info["lives"]
            self.n_steps = 0

        if self.start_with_fire:
            self.env.step(1)

        self.env.ale.getScreenGrayscale(self.screen_buffer[0])
        self.screen_buffer[1].fill(0)

        self.stacked_frames = deque(
            np.repeat(self.pool_and_resize()[None, ...], self.n_stacked_frames, axis=0),
            maxlen=self.n_stacked_frames,
        )
        self.state = np.array(self.stacked_frames)

        return self.state

    @partial(jax.jit, static_argnames="self")
    def is_absorbing(self, absorbing_: bool, info: dict, n_lives: int) -> Tuple[bool, int]:
        if self.terminal_on_life_loss:
            return jnp.logical_or(absorbing_, (info["lives"] < n_lives)), info["lives"]
        else:
            return absorbing_, info["lives"]

    def step(self, action: jnp.int8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        reward = 0

        for idx_frame in range(self.n_skipped_frames):
            _, reward_, absorbing_, _, info = self.env.step(action)

            reward += reward_
            absorbing, self.n_lives = self.is_absorbing(absorbing_, info, self.n_lives)

            if idx_frame >= self.n_skipped_frames - 2:
                t = idx_frame - (self.n_skipped_frames - 2)
                self.env.ale.getScreenGrayscale(self.screen_buffer[t])

            if absorbing:
                break

        self.stacked_frames.append(self.pool_and_resize())
        self.state = np.array(self.stacked_frames)

        self.n_steps += 1

        return self.state, np.array(np.clip(reward, -1, 1), dtype=np.int8), np.array(absorbing, dtype=np.bool_), _

    def pool_and_resize(self) -> np.ndarray:
        np.maximum(self.screen_buffer[0], self.screen_buffer[1], out=self.screen_buffer[0])

        return cv2.resize(
            self.screen_buffer[0], (self.state_width, self.state_height), interpolation=cv2.INTER_AREA
        ).astype(np.uint8)

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
        self.state = np.array(self.stacked_frames)
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
                self.reset(truncation=self.n_steps >= horizon)

    def collect_samples(
        self,
        q: BaseQ,
        q_params: hk.Params,
        n_samples: int,
        horizon: int,
        replay_buffer: ReplayBuffer,
        exploration_schedule: EpsilonGreedySchedule,
    ) -> None:
        for _ in range(n_samples):
            state = self.state

            if exploration_schedule.explore():
                action = q.random_action(exploration_schedule.key)
            else:
                action = q.best_action(exploration_schedule.key, q_params, state)

            next_state, reward, absorbing, _ = self.step(action)

            replay_buffer.add(state, action, reward, next_state, absorbing)

            if absorbing or self.n_steps >= horizon:
                self.reset(truncation=self.n_steps >= horizon)

    def evaluate_one_simulation(
        self,
        q: BaseQ,
        q_params: hk.Params,
        horizon: int,
        eps_eval: float,
        exploration_key: jax.random.PRNGKey,
        video_path: str,
        idx_head: int,
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
                action = q.random_action(key)
            else:
                if idx_head is None:
                    action = q.best_actions(q_params, self.state)[0]
                else:
                    action = q.best_actions(q_params, idx_head, self.state)[0]

            _, reward, absorbing, _ = self.step(action)

            cumulative_reward += discount * reward
            discount *= self.gamma

        if video_path is not None:
            video.close()
            os.remove(f"experiments/atari/figures/{video_path}.meta.json")

        return cumulative_reward

    def evaluate(
        self,
        q: BaseQ,
        q_params: hk.Params,
        horizon: int,
        n_simulations: int,
        eps_eval: float,
        exploration_key: jax.random.PRNGKey,
        video_path: str,
        idx_head: int = None,
    ) -> float:
        rewards = np.zeros(n_simulations)

        rewards[0] = self.evaluate_one_simulation(q, q_params, horizon, eps_eval, exploration_key, video_path, idx_head)
        for idx_simulation in range(1, n_simulations):
            rewards[idx_simulation] = self.evaluate_one_simulation(
                q, q_params, horizon, eps_eval, exploration_key, None, idx_head
            )

        return rewards.mean()
