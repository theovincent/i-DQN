import sys
import argparse
import json
import jax
import jax.numpy as jnp
from tqdm import tqdm


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Collect sample on Car-On-Hill.")
    parser.add_argument(
        "-e",
        "--experiment_name",
        help="Experiment name.",
        type=str,
        required=True,
    )
    args = parser.parse_args(argvs)
    print(f"{args.experiment_name}:")
    p = json.load(open(f"experiments/car_on_hill/figures/{args.experiment_name}/parameters.json"))  # p for parameters
    print(f"Collecting {p['n_samples']} samples on Car-On-Hill...")

    from experiments.car_on_hill.utils import define_environment
    from idqn.sample_collection.replay_buffer import ReplayBuffer

    sample_key = jax.random.PRNGKey(p["env_seed"])

    env = define_environment(p["gamma"], p["n_states_x"], p["n_states_v"])[0]

    replay_buffer = ReplayBuffer(p["n_samples"])

    env.reset()
    n_episodes = 0
    for _ in tqdm(range(p["n_samples"])):
        state = env.state

        sample_key, key = jax.random.split(sample_key)
        action = jax.random.choice(key, jnp.arange(env.n_actions))
        next_state, reward, absorbing, _ = env.step(action)

        replay_buffer.add(state, action, reward, next_state, absorbing)

        if absorbing or env.n_steps >= p["horizon"]:
            env.reset()
            n_episodes += 1

    assert sum(jnp.array(replay_buffer.rewards) == 1) > 0, "No positive reward has been sampled, please do something!"
    print(
        f"Number of episodes: {n_episodes}, number of positive reward: {sum(jnp.array(replay_buffer.rewards) == 1)}, number of negative reward: {sum(jnp.array(replay_buffer.rewards) == -1)}"
    )

    replay_buffer.save(f"experiments/car_on_hill/figures/{args.experiment_name}/replay_buffer.npz")
