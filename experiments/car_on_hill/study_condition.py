from tqdm import tqdm
from argparse import Namespace
import jax
import jax.numpy as jnp
import numpy as np
import json
import math

from idqn.sample_collection.replay_buffer import ReplayBuffer
from idqn.networks.architectures.ifqi import CarOnHilliFQI
from idqn.environments.car_on_hill import CarOnHillEnv
from idqn.utils.pickle import load_pickled_data
from idqn.sample_collection import IDX_RB

from jax import config

config.update("jax_enable_x64", True)

p = json.load(open(f"experiments/car_on_hill/parameters.json"))

env = CarOnHillEnv(p["gamma"])

replay_buffer = ReplayBuffer(
    (2,),
    p["replay_buffer_size"],
    p["batch_size"],
    p["n_step_return"],
    p["gamma"],
    lambda x: x,
    stack_size=1,
    observation_dtype=float,
)

q = CarOnHilliFQI(
    4 + 1,
    p["features"],
    (2,),
    env.n_actions,
    math.pow(p["gamma"], p["n_step_return"]),
    jax.random.PRNGKey(0),
    p["learning_rate"],
    p["optimizer_eps"],
)


env.collect_random_samples(
    jax.random.PRNGKey(0),
    replay_buffer,
    p["n_random_samples"],
    p["n_oriented_samples"],
    p["oriented_states"],
    p["horizon"],
)

n_training_steps = 0
dataset = replay_buffer.get_all_valid_samples()
states = env.grid_states(200)

q.params = load_pickled_data("experiments/car_on_hill/params_online_params")
q.target_params = load_pickled_data("experiments/car_on_hill/target_params_online_params")


# proposition_value = q.compute_proposition_value(q.params, q.target_params, dataset, states, p["gamma"])
# diff_approximation_errors = q.compute_diff_approximation_errors(q.params, q.target_params, dataset)

# # Condition = 0.0007757139, Approx error gain = -2.6624226e-05
# print(proposition_value, diff_approximation_errors)

# # Verify 1: new loss <= target shift + current_loss
# current_loss = (q.n_heads - 1) * q.loss_on_batch(q.params, q.target_params, dataset, None)
# new_loss = (q.n_heads - 1) * q.loss_on_batch(q.params, q.params, dataset, None)
# # shape (n_heads - 1, n_samples)
# previous_target_values = jax.vmap(q.compute_target, in_axes=(None, 0), out_axes=1)(q.target_params, dataset)
# current_target_values = jax.vmap(q.compute_target, in_axes=(None, 0), out_axes=1)(q.params, dataset)
# target_shift = np.sum(np.mean(np.square(previous_target_values - current_target_values), axis=1))
# print(
#     new_loss,
#     current_loss + target_shift,
#     new_loss <= current_loss + target_shift,
#     current_loss + target_shift - new_loss,
# )

# # Verify 2: target shift <= gamma shift
# shift = (
#     jax.vmap(q.compute_norm_2, in_axes=(None, None, 0), out_axes=1)(
#         jax.tree_map(lambda param: param[1:-1], q.params),
#         jax.tree_map(lambda param: param[1:-1], q.target_params),
#         states,
#     )
#     .max(axis=(1, 2))  # over the state-action pairs
#     .sum()  # over the heads
# )
# print(target_shift, p["gamma"] * shift, target_shift <= p["gamma"] * shift, p["gamma"] * shift - target_shift)


# # Verify 3: gamma shift + current_loss <= old_loss
# old_loss = (q.n_heads - 1) * q.loss_on_batch(q.target_params, q.target_params, dataset, None)

# print(
#     p["gamma"] * shift + current_loss,
#     old_loss,
#     p["gamma"] * shift + current_loss <= old_loss,
#     old_loss - p["gamma"] * shift - current_loss,
# )


# # Verify 4: new_loss <= old_loss
# print(new_loss, old_loss, new_loss <= old_loss, old_loss - new_loss)


# Verify 5: new loss <= target shift + current_loss
def triangular_ineq(sample, params, target_params):
    current_loss = q.loss(params, target_params, sample, None)
    print("current_loss", current_loss)
    new_loss = q.loss(params, params, sample, None)
    print("new_loss", new_loss)

    # shape (n_heads - 1, n_samples)
    previous_target_values = q.compute_target(target_params, sample)[0]
    current_target_values = q.compute_target(params, sample)[0]
    target_shift = np.square(previous_target_values - current_target_values)
    print("target_shift", target_shift)

    computed_current_values = q.apply(params, sample[IDX_RB["state"]])[0][sample[IDX_RB["action"]]]
    computed_previous_target_values = (
        sample[IDX_RB["reward"]] + p["gamma"] * q.apply(target_params, sample[IDX_RB["next_state"]])[0].max()
    )
    computed_current_target_values = (
        sample[IDX_RB["reward"]] + p["gamma"] * q.apply(params, sample[IDX_RB["next_state"]])[0].max()
    )

    computed_current_loss = np.sqrt(np.square(computed_previous_target_values - computed_current_values))
    computed_new_loss = np.sqrt(np.square(computed_current_target_values - computed_current_values))
    computed_target_shift = np.sqrt(np.square(computed_previous_target_values - computed_current_target_values))

    print("computed_current_loss", computed_current_loss)
    print("computed_new_loss", computed_new_loss)
    print("computed_target_shift", computed_target_shift)
    print(
        "computed_target_shift + computed_current_loss - computed_new_loss",
        computed_target_shift + computed_current_loss - computed_new_loss,
    )

    if new_loss > current_loss + target_shift:
        print(
            new_loss,
            current_loss + target_shift,
            new_loss <= current_loss + target_shift,
            current_loss + target_shift - new_loss,
        )
        print(sample)


for idx in tqdm(range(1)):
    for idx_head in [2]:
        triangular_ineq(
            jax.tree_map(lambda sample_: sample_[idx], dataset),
            jax.tree_map(lambda param: param[idx_head - 1 : idx_head + 1], q.params),
            jax.tree_map(lambda param: param[idx_head - 1 : idx_head + 1], q.target_params),
        )
