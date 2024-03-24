from tqdm import tqdm
from argparse import Namespace
import jax

from idqn.sample_collection.replay_buffer import ReplayBuffer
from idqn.networks.architectures.ifqi import CarOnHilliFQI
from idqn.environments.car_on_hill import CarOnHillEnv


def train(
    experiment_path: str,
    args: Namespace,
    p: dict,
    q: CarOnHilliFQI,
    env: CarOnHillEnv,
    replay_buffer: ReplayBuffer,
) -> None:
    n_target_update = p["n_bellman_iterations"] // args.bellman_iterations_scope
    n_training_steps_per_target_update = p["n_training_steps_per_bellman_iterations"] * args.bellman_iterations_scope

    env.collect_random_samples(
        jax.random.PRNGKey(0),
        replay_buffer,
        p["n_random_samples"],
        p["n_oriented_samples"],
        p["oriented_states"],
        p["horizon"],
    )

    for idx_target_update in tqdm(range(n_target_update)):
        for _ in range(n_training_steps_per_target_update):
            # 1 so that a training step is done and no rolling step is done
            q.update_online_params(1, replay_buffer)
            # 0 so that the target parameters are updated
            q.update_target_params(0)

        q.save(f"{experiment_path}{args.bellman_iterations_scope}_Q_s{args.seed}_t{idx_target_update}")
        q.rolling_step(q.params)
