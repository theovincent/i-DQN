from sys import set_asyncgen_hooks
import jax
import jax.numpy as jnp
from functools import partial
from flax.core import FrozenDict
from slim_idqn.sample_collection.replay_buffer import ReplayBuffer, TransitionElement


@partial(jax.jit, static_argnames=("best_action_fn", "n_actions", "epsilon_fn"))
def select_action(best_action_fn, params, state, key, n_actions, epsilon_fn, n_training_steps):
    uniform_key, action_key,network_selection_key = jax.random.split(key, 3)
    return jnp.where(
        jax.random.uniform(uniform_key) <= epsilon_fn(n_training_steps),  # if uniform < epsilon,
        jax.random.randint(action_key, (), 0, n_actions),  # take random action
        best_action_fn(params, state, network_selection_key),  # otherwise, take a greedy action
    )


def collect_single_sample(key, env, agent, rb: ReplayBuffer, p, epsilon_schedule, n_training_steps: int):

    action = select_action(
        agent.best_action,  agent.online_params, env.state, action_selection_key, env.n_actions, epsilon_schedule, n_training_steps
    ).item()

    obs = env.observation
    reward, absorbing = env.step(action)

    episode_end = absorbing or env.n_steps >= p["horizon"]
    rb.add(
        TransitionElement(
            observation=obs,
            action=action,
            reward=reward if rb._clipping is None else rb._clipping(reward),
            is_terminal=absorbing,
            episode_end=episode_end,
        )
    )

    if episode_end:
        env.reset()

    return reward, episode_end
