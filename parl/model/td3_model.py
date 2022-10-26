import numpy as np
import gym
from typing import List, Dict, Union, Optional


from .layers import FullyConnectedNetwork

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

torch, nn = try_import_torch()

# from ray.rllib.algorithms.ddpg.ddpg_torch_model


class TD3TorchModel(TorchModelV2, nn.Module):
    """Extension of standard TorchModelV2 for DDPG.

    Data flow:
        obs -> forward() -> model_out
        model_out -> get_policy_output() -> pi(s)
        model_out, actions -> get_q_values() -> Q(s, a)
        model_out, actions -> get_twin_q_values() -> Q_twin(s, a)

    Note that this class by itself is not a valid model unless you
    implement forward() in a subclass."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        # Extra DDPGActionModel args:
        actor_hiddens: Optional[List[int]] = None,
        actor_hidden_activation: str = "relu",
        critic_hiddens: Optional[List[int]] = None,
        critic_hidden_activation: str = "relu",
        twin_q: bool = False,
        add_layer_norm: bool = False,
    ):
        """Initialize variables of this model.

        Extra model kwargs:
            actor_hidden_activation: activation for actor network
            actor_hiddens: hidden layers sizes for actor network
            critic_hidden_activation: activation for critic network
            critic_hiddens: hidden layers sizes for critic network
            twin_q: build twin Q networks.
            add_layer_norm: Enable layer norm (for param noise).

        Note that the core layers for forward() are not defined here, this
        only defines the layers for the output heads. Those layers for
        forward() should be defined in subclasses of DDPGTorchModel.
        """
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        if actor_hiddens is None:
            actor_hiddens = [256, 256]

        if critic_hiddens is None:
            critic_hiddens = [256, 256]

        self.bounded = np.logical_and(
            self.action_space.bounded_above, self.action_space.bounded_below
        ).any()
        self.action_dim = np.product(self.action_space.shape)
        self.obs_dim = np.product(self.obs_space.shape)
        assert self.obs_dim == num_outputs

        # Build the policy network.
        self.policy_model = FullyConnectedNetwork(
            num_inputs=self.obs_dim,
            num_outputs=self.action_dim,
            hiddens=actor_hiddens,
            activation=actor_hidden_activation,
            output_activation="tanh" if self.bounded else None
        )

        # Build the Q-net(s), including target Q-net(s).
        self.q_model = FullyConnectedNetwork(
            num_inputs=self.obs_dim + self.action_dim,
            num_outputs=1,
            hiddens=critic_hiddens,
            activation=critic_hidden_activation
        )
        if twin_q:
            self.twin_q_model = FullyConnectedNetwork(
                num_inputs=self.obs_dim + self.action_dim,
                num_outputs=1,
                hiddens=critic_hiddens,
                activation=critic_hidden_activation
            )
        else:
            self.twin_q_model = None

    
    def forward(self, input_dict, state, seq_lens):
        return input_dict["obs_flat"].float(), state

    def get_q_values(self, model_out: TensorType, actions: TensorType) -> TensorType:
        """Return the Q estimates for the most recent forward pass.

        This implements Q(s, a).

        Args:
            model_out: obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].
            actions: Actions to return the Q-values for.
                Shape: [BATCH_SIZE, action_dim].

        Returns:
            tensor of shape [BATCH_SIZE].
        """
        return self.q_model(torch.cat([model_out, actions], -1))

    def get_twin_q_values(
        self, model_out: TensorType, actions: TensorType
    ) -> TensorType:
        """Same as get_q_values but using the twin Q net.

        This implements the twin Q(s, a).

        Args:
            model_out: obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].
            actions (Optional[Tensor]): Actions to return the Q-values for.
                Shape: [BATCH_SIZE, action_dim].

        Returns:
            tensor of shape [BATCH_SIZE].
        """
        return self.twin_q_model(torch.cat([model_out, actions], -1))

    def get_policy_output(self, model_out: TensorType) -> TensorType:
        """Return the action output for the most recent forward pass.

        This outputs the support for pi(s). For continuous action spaces, this
        is the action directly. For discrete, is is the mean / std dev.

        Args:
            model_out: obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].

        Returns:
            tensor of shape [BATCH_SIZE, action_out_size]
        """
        return self.policy_model(model_out)

    def policy_variables(
        self, as_dict: bool = False
    ) -> Union[List[TensorType], Dict[str, TensorType]]:
        """Return the list of variables for the policy net."""
        if as_dict:
            return self.policy_model.state_dict()
        return list(self.policy_model.parameters())

    def q_variables(
        self, as_dict=False
    ) -> Union[List[TensorType], Dict[str, TensorType]]:
        """Return the list of variables for Q / twin Q nets."""
        if as_dict:
            return {
                **self.q_model.state_dict(),
                **(self.twin_q_model.state_dict() if self.twin_q_model else {}),
            }
        return list(self.q_model.parameters()) + (
            list(self.twin_q_model.parameters()) if self.twin_q_model else []
        )


# Use sigmoid to scale to [0,1], but also double magnitude of input to
# emulate behaviour of tanh activation used in DDPG and TD3 papers.
# After sigmoid squashing, re-scale to env action space bounds.
# Note: when action_space=[-1,1], this op equals Tanh
class TanhSquash(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.low_action = nn.Parameter(
            torch.from_numpy(action_space.low).float(),
            requires_grad=False
        )

        self.action_range = nn.Parameter(
            torch.from_numpy(
                action_space.high - action_space.low
            ).float(),
            requires_grad=False
        )

    def forward(self, x):
        sigmoid_out = nn.Sigmoid()(2.0 * x)
        squashed = self.action_range * sigmoid_out + self.low_action
        return squashed
