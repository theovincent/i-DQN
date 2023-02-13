import jax.numpy as jnp


def importance_bound(gamma: float, n_bellman_iterations: int) -> jnp.ndarray:
    """
    $\alpha_k = (1 - \gamma) \gamma^{K-k-1} / (1 - \gamma^{K+1})$
    so
    $\alpha_k = (1 - \gamma) \gamma^{K-1} / ((1 - \gamma^{K+1}) \gamma^{k})$
    """

    pow_gammas = jnp.zeros(n_bellman_iterations)
    pow_gamma = 1
    for idx in range(n_bellman_iterations):
        pow_gammas = pow_gammas.at[idx].set(pow_gamma)
        pow_gamma *= gamma

    # multiply by n_bellman_iterations to avoid changing the learning rate
    return ((1 - gamma) * (pow_gamma / gamma)) / ((1 - pow_gamma * gamma) * pow_gammas) * n_bellman_iterations
