from typing import Union
import jax.numpy as jnp


def head_behaviorial_policy(method: Union[str, int], n_elements: int) -> jnp.ndarray:
    """
    Available methods: uniform | [int]
    """
    if method == "uniform":
        return jnp.ones(n_elements)
    elif type(method) != str:
        return jnp.zeros(n_elements).at[int(method)].set(1)
