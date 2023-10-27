import flax.linen as nn
import jax

from idqn.networks.dqn import DQN
from idqn.networks.architectures.base import Torso, Head


class AtariDQNNet(nn.Module):
    n_actions: int

    def setup(self):
        self.torso = Torso()
        self.head = Head(self.n_actions)

    def __call__(self, state):
        return self.head(self.torso(state))


class AtariDQN(DQN):
    def __init__(
        self,
        state_shape: list,
        n_actions: int,
        cumulative_gamma: float,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
        epsilon_optimizer: float,
        n_training_steps_per_online_update: int,
        n_training_steps_per_target_update: int,
    ) -> None:
        super().__init__(
            state_shape,
            n_actions,
            cumulative_gamma,
            AtariDQNNet(n_actions),
            network_key,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
        )
