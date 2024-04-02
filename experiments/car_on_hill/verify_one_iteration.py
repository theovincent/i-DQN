import json
import numpy as np
import math
import jax
import jax.numpy as jnp

from idqn.environments.car_on_hill import CarOnHillEnv
from idqn.networks.architectures.ifqi import CarOnHilliFQI
from idqn.sample_collection.replay_buffer import ReplayBuffer
from experiments.car_on_hill.iFQI_linear_closed_form import update_bellman_iteration


p = json.load(open(f"parameters.json"))

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
    2,
    "figures/data/features",
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
dataset = replay_buffer.get_all_valid_samples()


initial_params = jax.tree_map(lambda param: jnp.repeat(param[0][None], 2, axis=0), q.params)
q.params = initial_params
update_bellman_iteration(q, dataset)
optimal_params = q.params
optimal_weights = optimal_params["params"]["Dense_0"]["kernel"][0]

q.params = initial_params

optimal_loss = q.loss_on_batch(optimal_params, initial_params, dataset, None)

for _ in range(200):
    q.update_online_params(1, replay_buffer)

    weights = q.params["params"]["Dense_0"]["kernel"][1]

    print("||w* - w||", np.linalg.norm(weights - optimal_weights))
    loss = q.loss_on_batch(q.params, initial_params, dataset, None)
    if loss < optimal_loss:
        print("optimal loss has been beaten")
        print(loss)
        print(optimal_loss)

print(loss, optimal_loss)
grad = jax.grad(q.loss_on_batch)(q.params, initial_params, dataset, None)
print(jax.tree_map(lambda grad_: jnp.linalg.norm(grad_), grad))
print(grad)
print(weights)
print(optimal_weights)
