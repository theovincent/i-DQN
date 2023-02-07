# This file was inspired by https://github.com/MushroomRL/mushroom-rl

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
from scipy.integrate import odeint


from idqn.environments.viewer import Viewer
from idqn.networks.base_q import BaseQ, BaseMultiHeadQ


class CarOnHillEnv:
    """
    The Car On Hill selfironment as presented in:
    "Tree-Based Batch Mode Reinforcement Learning". Ernst D. et al.. 2005.
    """

    def __init__(self, gamma: float) -> None:
        self.gamma = gamma
        self.n_actions = 2
        self.max_position = 1.0
        self.max_velocity = 3.0
        self._g = 9.81
        self._m = 1.0
        self._dt = 0.1

        # Visualization
        self._viewer = Viewer(1, 1)

    def reset(self, state: jnp.ndarray = None) -> jnp.ndarray:
        if state is None:
            self.state = jnp.array([-0.5, 0])
        else:
            self.state = state

        self.n_steps = 0

        return self.state

    @partial(jax.jit, static_argnames="self")
    def boundery_conditions(self, new_state_odeint: jnp.ndarray) -> tuple:
        """
        A reward of -1 is obtained if the cart goes too fast or if the cart goes too far left,
        a reward of 1 if the cart goes too far right and not too fast otherwise the reward is 0.
        """
        state = jnp.array(new_state_odeint[-1, :-1])

        too_fast = (jnp.abs(state[1]) > self.max_velocity).astype(float)
        too_far_left = (state[0] < -self.max_position).astype(float)
        too_far_right = (self.max_position < state[0]).astype(float)

        too_far_left_or_too_fast = too_far_left + too_fast - too_far_left * too_fast
        too_far_right_and_not_too_fast = too_far_right * (1 - too_fast)

        reward, absorbing = too_far_left_or_too_fast * jnp.array(
            [-1.0, 1]
        ) + too_far_right_and_not_too_fast * jnp.array([1.0, 1])

        return state, reward, jnp.array(absorbing, dtype=bool)

    @partial(jax.jit, static_argnames="self")
    def append_state_action(self, state: jnp.ndarray, action: float):
        """action 0 maps to a force of -4 and action 1 maps to a force of 4."""
        return jnp.append(state, -4 + 8 * action)

    def step(self, action: float) -> tuple:
        new_state = odeint(self._dpds, self.append_state_action(self.state, action), [0, self._dt])

        self.n_steps += 1

        self.state, reward, absorbing = self.boundery_conditions(new_state)

        return self.state, reward, absorbing, {}

    def render(self, action: jnp.ndarray) -> None:
        # Slope
        self._viewer.function(0, 1, self._height)

        # Car
        car_body = [
            [-3e-2, 0],
            [-3e-2, 2e-2],
            [-2e-2, 2e-2],
            [-1e-2, 3e-2],
            [1e-2, 3e-2],
            [2e-2, 2e-2],
            [3e-2, 2e-2],
            [3e-2, 0],
        ]

        x_car = (np.array(self.state)[0] + 1) / 2
        y_car = self._height(x_car)
        c_car = [x_car, y_car]
        angle = self._angle(x_car)
        self._viewer.polygon(c_car, angle, car_body, color=(32, 193, 54))

        # Action
        self._viewer.force_arrow(c_car, np.array([action[0], 0]), 1, 20, 1, width=3)

        self._viewer.display(self._dt)

    @staticmethod
    def _angle(x):
        if x < 0.5:
            m = 4 * x - 1
        else:
            m = 1 / ((20 * x**2 - 20 * x + 6) ** 1.5)

        return np.arctan(m)

    @staticmethod
    def _height(x):
        y_neg = 4 * x**2 - 2 * x
        y_pos = (2 * x - 1) / np.sqrt(5 * (2 * x - 1) ** 2 + 1)
        y = np.zeros_like(x)

        mask = x < 0.5
        neg_mask = np.logical_not(mask)
        y[mask] = y_neg[mask]
        y[neg_mask] = y_pos[neg_mask]

        y_norm = (y + 1) / 2

        return y_norm

    def _dpds(self, state_action, t):
        position = state_action[0]
        velocity = state_action[1]
        u = state_action[-1]

        if position < 0.0:
            diff_hill = 2 * position + 1
            diff_2_hill = 2
        else:
            diff_hill = 1 / ((1 + 5 * position**2) ** 1.5)
            diff_2_hill = (-15 * position) / ((1 + 5 * position**2) ** 2.5)

        dp = velocity
        ds = (u - self._g * self._m * diff_hill - velocity**2 * self._m * diff_hill * diff_2_hill) / (
            self._m * (1 + diff_hill**2)
        )

        return dp, ds, 0.0

    def close(self):
        return self._viewer.close()

    def optimal_steps_to_absorbing(self, state: jnp.ndarray, max_steps: int) -> tuple:
        current_states = [state]
        step = 0

        while len(current_states) > 0 and step < max_steps:
            next_states = []
            for state_ in current_states:
                for idx_action in range(2):
                    self.state = state_
                    next_state, reward, _, _ = self.step(idx_action)

                    if reward == 1:
                        return True, step + 1
                    elif reward == 0:
                        next_states.append(next_state)
                    ## if reward == -1 we pass

            step += 1
            current_states = next_states

        return False, step

    def optimal_v(self, state: jnp.ndarray, max_steps: int) -> float:
        success, step_to_absorbing = self.optimal_steps_to_absorbing(state, max_steps)

        if step_to_absorbing == 0:
            return 0
        else:
            return self.gamma ** (step_to_absorbing - 1) if success else -self.gamma ** (step_to_absorbing - 1)

    @partial(jax.jit, static_argnames=("self", "q"))
    def q_estimate_mesh(self, q: BaseQ, params: hk.Params, states_x: jnp.ndarray, states_v: jnp.ndarray) -> jnp.ndarray:
        n_boxes = states_x.shape[0] * states_v.shape[0]
        states_x_mesh, states_v_mesh = jnp.meshgrid(states_x, states_v, indexing="ij")

        states = jnp.hstack((states_x_mesh.reshape((n_boxes, 1)), states_v_mesh.reshape((n_boxes, 1))))

        # Dangerous reshape: the indexing of meshgrid is 'ij'.
        return q(params, states).reshape((states_x.shape[0], states_v.shape[0], self.n_actions))

    @partial(jax.jit, static_argnames=("self", "q"))
    def q_multi_head_estimate_mesh(
        self, q: BaseMultiHeadQ, idx_head: int, params: hk.Params, states_x: jnp.ndarray, states_v: jnp.ndarray
    ) -> jnp.ndarray:
        n_boxes = states_x.shape[0] * states_v.shape[0]
        states_x_mesh, states_v_mesh = jnp.meshgrid(states_x, states_v, indexing="ij")

        states = jnp.hstack((states_x_mesh.reshape((n_boxes, 1)), states_v_mesh.reshape((n_boxes, 1))))

        # Dangerous reshape: the indexing of meshgrid is 'ij'.
        return q(params, states).reshape((states_x.shape[0], states_v.shape[0], q.n_heads, self.n_actions))[
            :, :, idx_head
        ]

    def evaluate(
        self, q: BaseQ, params: hk.Params, horizon: int, initial_state: jnp.ndarray, render: bool = False
    ) -> float:
        performance = 0
        discount = 1
        self.reset(initial_state)
        absorbing = False

        while not absorbing and self.n_steps < horizon:
            action = q(params, self.state).argmax()
            _, reward, absorbing, _ = self.step(action)

            if render:
                self.render(action)
            performance += discount * reward
            discount *= self.gamma

        self.close()

        return performance

    def evaluate_multi_head(
        self,
        q: BaseMultiHeadQ,
        idx_head: int,
        params: hk.Params,
        horizon: int,
        initial_state: jnp.ndarray,
        render: bool = False,
    ) -> float:
        performance = 0
        discount = 1
        self.reset(initial_state)
        absorbing = False

        while not absorbing and self.n_steps < horizon:
            action = q(params, self.state)[0, idx_head].argmax()
            _, reward, absorbing, _ = self.step(action)

            if render:
                self.render(action)
            performance += discount * reward
            discount *= self.gamma

        self.close()

        return performance

    def v_mesh(
        self, q: BaseQ, params: hk.Params, horizon: int, states_x: jnp.ndarray, states_v: jnp.ndarray
    ) -> np.ndarray:
        v_mesh_ = np.zeros((len(states_x), len(states_v)))

        for idx_state_x, state_x in enumerate(states_x):
            for idx_state_v, state_v in enumerate(states_v):
                v_mesh_[idx_state_x, idx_state_v] = self.evaluate(q, params, horizon, jnp.array([state_x, state_v]))

        return v_mesh_

    def v_mesh_multi_head(
        self,
        q: BaseMultiHeadQ,
        idx_head: int,
        params: hk.Params,
        horizon: int,
        states_x: jnp.ndarray,
        states_v: jnp.ndarray,
    ) -> np.ndarray:
        v_mesh_ = np.zeros((len(states_x), len(states_v)))

        for idx_state_x, state_x in enumerate(states_x):
            for idx_state_v, state_v in enumerate(states_v):
                v_mesh_[idx_state_x, idx_state_v] = self.evaluate_multi_head(
                    q, idx_head, params, horizon, jnp.array([state_x, state_v])
                )

        return v_mesh_
