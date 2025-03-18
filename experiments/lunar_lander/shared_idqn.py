import os
import sys

import jax

from experiments.base.shared_idqn import train
from experiments.base.utils import prepare_logs
from slim_idqn.environments.lunar_lander import LunarLander
from slim_idqn.networks.shared_idqn import SharedLayeriDQN
from slim_idqn.sample_collection.replay_buffer import ReplayBuffer
from slim_idqn.sample_collection.samplers import UniformSamplingDistribution


def run(argvs=sys.argv[1:]):
    env_name, algo_name = os.path.abspath(__file__).split("/")[-2], os.path.abspath(__file__).split("/")[-1][:-3]
    p = prepare_logs(env_name, algo_name, argvs)

    q_key, train_key = jax.random.split(jax.random.PRNGKey(p["seed"]))

    env = LunarLander()
    rb = ReplayBuffer(
        sampling_distribution=UniformSamplingDistribution(p["seed"]),
        batch_size=p["batch_size"],
        max_capacity=p["replay_buffer_capacity"],
        stack_size=1,
        update_horizon=p["update_horizon"],
        gamma=p["gamma"],
        compress=True,
    )
    agent = SharedLayeriDQN(
        q_key,
        env.observation_shape[0],
        env.n_actions,
        features=p["features"],
        learning_rate=p["learning_rate"],
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        update_to_data=p["update_to_data"],
        target_update_frequency=p["target_update_frequency"],
        shift_params_frequency=p["shift_params_frequency"],
        adam_eps=1.5e-4,
        num_networks=p["num_of_networks"],
        num_shared_layers = p["num_shared_layers"]
    ) 
    train(train_key, p, agent, env, rb)


if __name__ == "__main__":
    run()
