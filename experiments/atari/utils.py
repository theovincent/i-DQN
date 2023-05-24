from jax.random import KeyArray
import jax


def generate_keys(seed: int) -> KeyArray:
    return jax.random.split(jax.random.PRNGKey(seed))
