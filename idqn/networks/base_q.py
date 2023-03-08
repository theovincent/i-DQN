from functools import partial
import haiku as hk
import optax
import jax
import jax.numpy as jnp


class BaseQ:
    def __init__(
        self,
        state_shape: list,
        gamma: float,
        network: hk.Module,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
    ) -> None:
        self.state_shape = state_shape
        self.gamma = gamma
        self.network = hk.without_apply_rng(hk.transform(network))
        self.network_key = network_key
        self.params = self.network.init(rng=self.network_key, state=jnp.zeros(self.state_shape, dtype=jnp.float32))

        self.loss_and_grad = jax.jit(jax.value_and_grad(self.loss))

        if learning_rate is not None:
            self.learning_rate = learning_rate
            self.optimizer = optax.adam(self.learning_rate, eps=0.0003125)
            self.optimizer_state = self.optimizer.init(self.params)

    def random_init_params(self) -> hk.Params:
        self.network_key, key = jax.random.split(self.network_key)

        return self.random_init_params_(key)

    @partial(jax.jit, static_argnames="self")
    def random_init_params_(self, key: jax.random.PRNGKeyArray) -> hk.Params:
        return self.network.init(rng=key, state=jnp.zeros(self.state_shape))

    @partial(jax.jit, static_argnames="self")
    def __call__(self, params: hk.Params, states: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply(params, states)

    def loss(self, params: hk.Params, params_target: hk.Params, samples: dict, ord: int = 2) -> jnp.ndarray:
        raise NotImplementedError

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self, params: hk.Params, params_target: hk.Params, optimizer_state: tuple, batch_samples: jnp.ndarray
    ) -> tuple:
        loss, grad_loss = self.loss_and_grad(params, params_target, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    def reset_optimizer(self) -> None:
        self.optimizer = optax.adam(self.learning_rate)
        self.optimizer_state = self.optimizer.init(self.params)


class BaseMultiHeadQ(BaseQ):
    def __init__(
        self,
        n_heads: int,
        state_shape: list,
        gamma: float,
        network: hk.Module,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
    ) -> None:
        self.n_heads = n_heads
        super().__init__(
            state_shape=state_shape,
            gamma=gamma,
            network=network,
            network_key=network_key,
            learning_rate=learning_rate,
        )

    @partial(jax.jit, static_argnames="self")
    def compute_target(self, params: hk.Params, samples: dict) -> jnp.ndarray:
        return jnp.repeat(samples["reward"][:, None], self.n_heads, axis=1) + jnp.repeat(
            1 - samples["absorbing"][:, None], self.n_heads, axis=1
        ) * self.gamma * self(params, samples["next_state"]).max(axis=2)


class iQ(BaseMultiHeadQ):
    def __init__(
        self,
        importance_iteration: jnp.ndarray,
        state_shape: list,
        gamma: float,
        network: hk.Module,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
    ) -> None:
        self.importance_iteration = importance_iteration

        super().__init__(
            n_heads=len(importance_iteration) + 1,
            state_shape=state_shape,
            gamma=gamma,
            network=network,
            network_key=network_key,
            learning_rate=learning_rate,
        )

    def move_forward(self, params: hk.Params) -> hk.Params:
        raise NotImplementedError

    @partial(jax.jit, static_argnames=("self", "ord"))
    def loss(self, params: hk.Params, params_target: hk.Params, samples: dict, ord: str = "huber") -> jnp.ndarray:
        targets = self.compute_target(params_target, samples)[:, :-1]
        predictions = self(params, samples["state"])[jnp.arange(samples["state"].shape[0]), 1:, samples["action"]]

        error = (predictions - targets) * jnp.repeat(self.importance_iteration[None, :], targets.shape[0], axis=0)
        if ord == "huber":
            return optax.huber_loss(error, 0).mean()
        elif ord == "sum":
            return jnp.square(error).sum()

    @partial(jax.jit, static_argnames=("self", "ord"))
    def bellman_errors(self, params: hk.Params, params_target: hk.Params, samples: dict, ord: str = "2") -> jnp.ndarray:
        targets = self.compute_target(params_target, samples)
        predictions = self(params, samples["state"])[jnp.arange(samples["state"].shape[0]), :, samples["action"]]

        error = predictions - targets
        if ord == "1":
            return jnp.abs(error).mean(axis=0)
        elif ord == "2":
            return jnp.square(error).mean(axis=0)
        elif ord == "sum":
            return jnp.square(error).sum(axis=0)
