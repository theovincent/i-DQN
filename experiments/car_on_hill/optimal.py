import os


def run_optimal():
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)
    import jax.numpy as jnp
    import json

    parameters = json.load(open("parameters.json"))
    gamma = parameters["gamma"]

    # Visualisation of errors and performances
    n_states_x = parameters["n_states_x"]
    n_states_v = parameters["n_states_v"]
    horizon = parameters["horizon"]

    ## Define the environment
    import numpy as np

    from idqn.environments.car_on_hill import CarOnHillEnv

    env = CarOnHillEnv(gamma)

    states_x = jnp.linspace(-env.max_position, env.max_position, n_states_x)
    states_v = jnp.linspace(-env.max_velocity, env.max_velocity, n_states_v)

    ## Compute the optimal value function
    from tqdm import tqdm

    optimal_v = np.zeros((n_states_x, n_states_v))

    for idx_state_x, state_x in enumerate(tqdm(states_x)):
        for idx_state_v, state_v in enumerate(tqdm(states_v, leave=False)):
            optimal_v[idx_state_x, idx_state_v] = env.optimal_v(jnp.array([state_x, state_v]), horizon)

    np.save(f"figures/data/optimal/V.npy", optimal_v)

    ## Compute the optimal state-action value function
    optimal_q = np.zeros((n_states_x, n_states_v, 2))

    for idx_state_x, state_x in enumerate(tqdm(states_x)):
        for idx_state_v, state_v in enumerate(tqdm(states_v, leave=False)):
            for idx_action in range(2):
                env.reset(np.array([state_x, state_v]))
                next_state, reward, absorbing, _ = env.step(idx_action)

                if absorbing:
                    optimal_q[idx_state_x, idx_state_v, idx_action] = reward
                else:
                    optimal_q[idx_state_x, idx_state_v, idx_action] = reward + gamma * env.optimal_v(
                        next_state, horizon
                    )

    if not os.path.exists("figures/data/optimal/"):
        os.makedirs("figures/data/optimal/")
    np.save(f"figures/data/optimal/Q.npy", optimal_q)


if __name__ == "__main__":
    run_optimal()
