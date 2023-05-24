import json
import jax
from idqn.utils.pickle import load_pickled_data
from idqn.utils.importance_iteration import importance_iteration
from idqn.environments.atari import AtariEnv
from idqn.networks.q_architectures import AtariDQN, AtariiDQN


# ------- To modify ------- #
experiment = "ut30_uh6000"
algorithm = "iDQN"
game = "Hero"
bellman_iterations_scope = 10
parameters = "Q_11_194_best"
# ------------------------- #

if algorithm == "DQN":
    params_path = f"{experiment}/{game}/{algorithm}/{parameters}"
else:
    params_path = f"{experiment}/{game}/{algorithm}/{bellman_iterations_scope}_{parameters}"

p = json.load(open(f"experiments/atari/figures/{experiment}/parameters.json"))

env = AtariEnv(game)

if algorithm == "DQN":
    q = AtariDQN(
        (env.n_stacked_frames, env.state_height, env.state_width),
        env.n_actions,
        p["gamma"],
        jax.random.PRNGKey(0),
        None,
        None,
        None,
    )
else:
    q = AtariiDQN(
        importance_iteration(p["idqn_importance_iteration"], p["gamma"], bellman_iterations_scope),
        (env.n_stacked_frames, env.state_height, env.state_width),
        env.n_actions,
        p["gamma"],
        jax.random.PRNGKey(0),
        importance_iteration(p["idqn_head_behaviorial_policy"], p["gamma"], bellman_iterations_scope + 1),
        None,
        None,
        None,
        None,
    )

q_params = load_pickled_data(f"experiments/atari/figures/{params_path}_online_params")


reward = env.evaluate_one_simulation(
    q,
    q_params,
    p["horizon"],
    p["ending_eps"],
    jax.random.PRNGKey(0),
    params_path,
)
print("Reward:", reward)
