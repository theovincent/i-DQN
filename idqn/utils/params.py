import pickle
import jax
import haiku as hk


def save_params(path: str, params: hk.Params) -> None:
    params = jax.device_get(params)
    with open(path, "wb") as fp:
        pickle.dump(params, fp)


def load_params(path: str) -> hk.Params:
    with open(path, "rb") as fp:
        params = pickle.load(fp)
    return jax.device_put(params)


def add_values(layers_param_1: dict, layers_param_2: dict) -> dict:
    sum_params = {}
    for key_weight_layer in layers_param_1.keys():
        sum_params[key_weight_layer] = layers_param_1[key_weight_layer] + layers_param_2[key_weight_layer]

    return sum_params
