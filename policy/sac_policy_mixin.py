from ray.rllib.utils.typing import ModelWeights
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.numpy import convert_to_numpy

import torch
import numpy as np


class SACEvolveMixin:
    """
        Add two methods to TorchPolicy for NeuroEvolution
    """
    def __init__(self):
        # Note: apply this at `before_loss_init`
        self.params_shape = {}
        self.num_evolve_params = 0

        for name, param in self.model.action_model.named_parameters():
            self.params_shape[name] = param.size()
            self.num_evolve_params += param.numel()

    def get_evolution_weights(self) -> ModelWeights:
        # only need learnable weights in policy(actor)
        state_dict = dict(self.model.action_model.named_parameters())

        return convert_to_numpy(state_dict)

    def set_evolution_weights(self, weights: ModelWeights):
        state_dict = convert_to_torch_tensor(weights)

        self.model.action_model.load_state_dict(state_dict, strict=False)
