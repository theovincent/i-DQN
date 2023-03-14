import pickle
import jax


def save_pickled_data(path: str, object):
    object = jax.device_get(object)

    with open(path, "wb") as handle:
        pickle.dump(object, handle)


def load_pickled_data(path: str):
    with open(path, "rb") as handle:
        object = pickle.load(handle)

    return jax.device_put(object)
