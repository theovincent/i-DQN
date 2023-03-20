import json
import jax
from idqn.utils.pickle import load_pickled_data
from idqn.environments.atari import AtariEnv
from idqn.networks.q_architectures import AtariDQN

params_path = "DQN_5/Breakout/DQN/Q_1_99"

p = json.load(open(f"experiments/atari/figures/{params_path.split('/')[0]}/parameters.json"))

env = AtariEnv(params_path.split("/")[1])

q = AtariDQN(
    (env.n_stacked_frames, env.state_height, env.state_width),
    env.n_actions,
    p["gamma"],
    jax.random.PRNGKey(0),
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
