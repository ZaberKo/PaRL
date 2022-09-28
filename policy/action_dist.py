
import torch.nn.functional as F
import gym
import numpy as np

from ray.rllib.models.torch.torch_action_dist import TorchSquashedGaussian

from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import SMALL_NUMBER, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT
from ray.rllib.utils.typing import TensorType, List, Union, Tuple, ModelConfigDict

torch, nn = try_import_torch()

# numerically stable version of TorchSquashedGaussian


class SquashedGaussian(TorchSquashedGaussian):
    @override(TorchSquashedGaussian)
    def logp(self, x: TensorType) -> TensorType:
        # Unsquash values (from [low,high] to ]-inf,inf[)
        unsquashed_values = self._unsquash(x)  # means `u` in SAC paper
        # For safety reasons, clamp somehow, only then sum up.
        log_prob_gaussian = torch.clamp(self.dist.log_prob(
            unsquashed_values), -100, 100).sum(dim=-1)

        # Note: use magic code from Spinning-up repo
        log_prob = log_prob_gaussian - torch.sum(
            2*np.log(2)-unsquashed_values-F.softplus(-2*unsquashed_values), 
            dim=-1)
        return log_prob

    def sample_logp(self):
        z = self.dist.rsample()
        actions = self._squash(z)


        # For safety reasons, clamp somehow, only then sum up.
        log_prob_gaussian = torch.clamp(self.dist.log_prob(z), -100, 100).sum(dim=-1)
        # Note: use magic code from Spinning-up repo
        log_prob = log_prob_gaussian - torch.sum(
            2*np.log(2)-z-F.softplus(-2*z), 
            dim=-1)
        return 
