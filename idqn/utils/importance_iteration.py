import jax.numpy as jnp


def importance_iteration(method: str, gamma: float, n_elements: int) -> jnp.ndarray:
    """
    Available methods: bound | uniform | last
    For method = bound:
        $\alpha_k = (1 - \gamma) \gamma^{K-k-1} / (1 - \gamma^{K+1})$
        so
        $\alpha_k = (1 - \gamma) \gamma^{K-1} / ((1 - \gamma^{K+1}) \gamma^{k})$
    """
    if method == "bound":
        pow_gammas = jnp.zeros(n_elements)
        pow_gamma = 1
        for idx in range(n_elements):
            pow_gammas = pow_gammas.at[idx].set(pow_gamma)
            pow_gamma *= gamma

        importance = ((1 - gamma) * (pow_gamma / gamma)) / ((1 - pow_gamma * gamma) * pow_gammas)

        return importance / importance.sum() * n_elements
    elif method == "uniform":
        return jnp.ones(n_elements)
    elif method == "last":
        return jnp.zeros(n_elements).at[-1].set(1)
