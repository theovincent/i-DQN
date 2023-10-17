import json
import jax
from idqn.utils.pickle import load_pickled_data
from idqn.utils.head_behaviorial_policy import head_behaviorial_policy
from idqn.environments.atari import AtariEnv
from idqn.networks.q_architectures import AtariDQN, AtariIQN, AtariiDQN


# ------- To modify ------- #
experiment = "ut30_uh6000"
algorithm = "iDQN"
game = "ChopperCommand"
bellman_iterations_scope = 5
parameters = "Q_12_199_best"
# ------------------------- #

if bellman_iterations_scope is None:
    params_path = f"{experiment}/{game}/{algorithm}/{parameters}"
else:
    params_path = f"{experiment}/{game}/{algorithm}/{bellman_iterations_scope}_{parameters}"

p = json.load(open(f"experiments/atari/figures/{experiment}/parameters.json"))

env = AtariEnv(game)

if algorithm == "DQN":
    q = AtariDQN(
        (env.state_height, env.state_width, env.n_stacked_frames),
        env.n_actions,
        p["gamma"],
        jax.random.PRNGKey(0),
        None,
        None,
        None,
        None,
    )
elif algorithm == "IQN":
    q = AtariIQN(
        (env.state_height, env.state_width, env.n_stacked_frames),
        env.n_actions,
        p["gamma"],
        jax.random.PRNGKey(0),
        None,
        None,
        None,
        None,
    )
else:
    q = AtariiDQN(
        bellman_iterations_scope + 1,
        (env.state_height, env.state_width, env.n_stacked_frames),
        env.n_actions,
        p["gamma"],
        jax.random.PRNGKey(0),
        head_behaviorial_policy(p["idqn_head_behaviorial_policy"], bellman_iterations_scope + 1),
        None,
        None,
        None,
        None,
        None,
    )

q_params = load_pickled_data(f"experiments/atari/figures/{params_path}_online_params")


reward, absorbing = env.evaluate_one_simulation(
    q, q_params, p["horizon"], p["ending_eps"], jax.random.PRNGKey(0), params_path
)
print("Reward:", reward)
print("N steps", env.n_steps, "; Horizong", p["horizon"], "; Absorbing", absorbing)
