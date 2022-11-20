import numpy as np
import gym

from ray.rllib.models.utils import get_activation_fn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

from typing import Dict, List, Optional, Tuple, Union
from ray.rllib.utils.typing import ModelConfigDict, TensorType, TensorStructType

torch, nn = try_import_torch()


class FullyConnectedNetwork(nn.Module):
    """Generic fully connected network."""

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        hiddens: List[int],
        activation='relu',
        output_activation=None,
        add_layer_norm: bool = False,
    ):
        super(FullyConnectedNetwork, self).__init__()

        self._hidden_layers = nn.Sequential()

        prev_layer_size = num_inputs

        # Create layers 0 to second-last.
        for i, size in enumerate(hiddens):
            self._hidden_layers.add_module(
                f"layer_{i}",
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    activation_fn=activation,
                    use_bias=True,
                    add_layer_norm=add_layer_norm
                )
            )
            prev_layer_size = size

        # output layer
        self._logits = SlimFC(
            in_size=prev_layer_size,
            out_size=num_outputs,
            activation_fn=output_activation,
            use_bias=True,
            add_layer_norm=False
        )

    def forward(self, x):
        features = self._hidden_layers(x)
        logits = self._logits(features)

        return logits


class SlimFC(nn.Module):
    """Simple PyTorch version of `linear` function"""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        activation_fn=None,
        use_bias: bool = True,
        add_layer_norm=False,
    ):
        """Creates a standard FC layer, similar to torch.nn.Linear

        Args:
            in_size(int): Input size for FC Layer
            out_size: Output size for FC Layer
            initializer: Initializer function for FC layer weights
            activation_fn: Activation function at the end of layer
            use_bias: Whether to add bias weights or not
            bias_init: Initalize bias weights to bias_init const
        """
        super(SlimFC, self).__init__()
        self._model = nn.Sequential()

        self._model.add_module(
            "linear",
            nn.Linear(in_size, out_size, bias=use_bias)
        )

        if add_layer_norm:
            self._model.add_module(
                "layer_norm",
                nn.LayerNorm(out_size, elementwise_affine=False)
            )

        if isinstance(activation_fn, str):
            activation_fn = get_activation_fn(activation_fn, "torch")

        if activation_fn is not None:
            self._model.add_module(
                "activation_fn",
                activation_fn()
                )

    def forward(self, x: TensorType) -> TensorType:
        return self._model(x)
