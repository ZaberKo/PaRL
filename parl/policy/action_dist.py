
import torch.nn.functional as F
import gym
import numpy as np

from ray.rllib.models.torch.torch_action_dist import TorchSquashedGaussian, TorchDistributionWrapper

from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
# from ray.rllib.utils.numpy import (
#     SMALL_NUMBER,
#     MIN_LOG_NN_OUTPUT,
#     MAX_LOG_NN_OUTPUT
# )
from ray.rllib.utils.typing import TensorType, List, Union, Tuple, ModelConfigDict

torch, nn = try_import_torch()



SMALL_NUMBER=1e-9
MIN_LOG_NN_OUTPUT = -20
MAX_LOG_NN_OUTPUT = 2

# numerically stable version of TorchSquashedGaussian
class SquashedGaussian(TorchDistributionWrapper):
    def __init__(
        self,
        inputs: List[TensorType],
        model: TorchModelV2,
        low: float = -1.0,
        high: float = 1.0,
    ):
        """Parameterizes the distribution via `inputs`.

        Args:
            low: The lowest possible sampling value
                (excluding this value).
            high: The highest possible sampling value
                (excluding this value).
        """
        super().__init__(inputs, model)
        # Split inputs into mean and log(std).
        mean, log_std = torch.chunk(self.inputs, 2, dim=-1)
        # Clip `scale` values (coming from NN) to reasonable values.
        log_std = torch.clamp(log_std, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT)
        std = torch.exp(log_std)
        self.dist = torch.distributions.normal.Normal(mean, std)
        assert np.all(np.less(low, high))
        self.low = low
        self.high = high
        self.mean = mean
        self.std = std

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        self.last_sample = self._squash(self.dist.mean)
        return self.last_sample

    @override(TorchDistributionWrapper)
    def sample(self) -> TensorType:
        # Use the reparameterization version of `dist.sample` to allow for
        # the results to be backprop'able e.g. in a loss term.

        normal_sample = self.dist.rsample()
        self.last_sample = self._squash(normal_sample)
        return self.last_sample

    @override(TorchDistributionWrapper)
    def logp(self, x: TensorType) -> TensorType:
        # Unsquash values (from [low,high] to ]-inf,inf[)
        u = self._unsquash(x)  # means `u` in SAC paper
        # For safety reasons, clamp somehow, only then sum up.
        log_prob_gaussian = torch.clamp(self.dist.log_prob(
            u), -100, 100).sum(dim=-1)

        # Note: use magic code from Spinning-up repo
        log_prob = log_prob_gaussian - 2*torch.sum(
            np.log(2)-u-F.softplus(-2*u),
            dim=-1)
        return log_prob

    def sample_logp(self):
        z = self.dist.rsample()
        actions = self._squash(z)

        # For safety reasons, clamp somehow, only then sum up.
        log_prob_gaussian = torch.clamp(
            self.dist.log_prob(z), -100, 100).sum(dim=-1)
        # Note: use magic code from Spinning-up repo
        log_prob = log_prob_gaussian - 2*torch.sum(
            np.log(2)-z-F.softplus(-2*z), dim=-1)
        return actions, log_prob

    def _squash(self, raw_values: TensorType) -> TensorType:
        # Returned values are within [low, high] (including `low` and `high`).
        squashed = ((torch.tanh(raw_values) + 1.0) / 2.0) * (
            self.high - self.low
        ) + self.low
        return torch.clamp(squashed, self.low, self.high)

    def _unsquash(self, values: TensorType) -> TensorType:
        normed_values = (values - self.low) / (self.high - self.low) * 2.0 - 1.0
        # Stabilize input to atanh.
        save_normed_values = torch.clamp(
            normed_values, -1.0 + SMALL_NUMBER, 1.0 - SMALL_NUMBER
        )
        unsquashed = torch.atanh(save_normed_values)
        return unsquashed

    def sampled_action_logp(self) -> TensorType:
        assert self.last_sample is not None
        return self.logp(self.last_sample)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(
        action_space: gym.Space, model_config: ModelConfigDict
    ) -> Union[int, np.ndarray]:
        return np.prod(action_space.shape, dtype=np.int32) * 2
    
    @override(TorchDistributionWrapper)
    def entropy(self) -> TensorType:
        raise ValueError("Entropy not defined for SquashedGaussian!")

    @override(TorchDistributionWrapper)
    def kl(self, other: ActionDistribution) -> TensorType:
        raise ValueError("KL not defined for SquashedGaussian!")

class SquashedGaussian2(TorchDistributionWrapper):
    """
        only output action [-1, 1], following spinup version
    """

    def __init__(
        self,
        inputs: List[TensorType],
        model: TorchModelV2,
    ):
        """Parameterizes the distribution via `inputs`.

        Args:
            low: The lowest possible sampling value
                (excluding this value).
            high: The highest possible sampling value
                (excluding this value).
        """
        super().__init__(inputs, model)
        # Split inputs into mean and log(std).
        mean, log_std = torch.chunk(self.inputs, 2, dim=-1) # [B,n]
        # Clip `scale` values (coming from NN) to reasonable values.
        log_std = torch.clamp(log_std, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT)
        std = torch.exp(log_std)
        self.dist = torch.distributions.normal.Normal(mean, std)

        self.mean = mean
        self.std = std

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        self.last_sample = self._squash(self.dist.mean)
        return self.last_sample

    @override(TorchDistributionWrapper)
    def sample(self) -> TensorType:
        # Use the reparameterization version of `dist.sample` to allow for
        # the results to be backprop'able e.g. in a loss term.

        normal_sample = self.dist.rsample()
        self.last_sample = self._squash(normal_sample)
        return self.last_sample

    @override(ActionDistribution)
    def logp(self, x: TensorType) -> TensorType:
        # Unsquash values (from [low,high] to ]-inf,inf[)
        z = self._unsquash(x)
        # For safety reasons, clamp somehow, only then sum up.
        log_prob_gaussian = self.dist.log_prob(z).sum(dim=-1)

        # Note: use magic code from Spinning-up repo
        log_prob = log_prob_gaussian - 2*torch.sum(
            np.log(2)-z-F.softplus(-2*z),
            dim=-1)
        return log_prob

    def sample_logp(self):
        z = self.dist.rsample()
        actions = self._squash(z)

        # For safety reasons, clamp somehow, only then sum up.
        log_prob_gaussian = self.dist.log_prob(z).sum(dim=-1)
        # Note: use magic code from Spinning-up repo
        log_prob = log_prob_gaussian - 2*torch.sum(
            np.log(2)-z-F.softplus(-2*z), dim=-1)
        return actions, log_prob

    @override(TorchDistributionWrapper)
    def entropy(self) -> TensorType:
        raise ValueError("Entropy not defined for SquashedGaussian!")

    @override(TorchDistributionWrapper)
    def kl(self, other: ActionDistribution) -> TensorType:
        raise ValueError("KL not defined for SquashedGaussian!")

    def _squash(self, raw_values: TensorType) -> TensorType:
        return torch.tanh(raw_values)

    def _unsquash(self, values: TensorType) -> TensorType:
        # Stabilize input to atanh.
        save_normed_values = torch.clamp(
            values, -1.0, 1.0
        )
        unsquashed = torch.atanh(save_normed_values)
        return unsquashed

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(
        action_space: gym.Space, model_config: ModelConfigDict
    ) -> Union[int, np.ndarray]:
        return np.prod(action_space.shape, dtype=np.int32) * 2


class SquashedGaussian3(TorchDistributionWrapper):
    """
        only output action [-1, 1], implemented by native torch.distribution
    """

    def __init__(
        self,
        inputs: List[TensorType],
        model: TorchModelV2,
    ):
        """Parameterizes the distribution via `inputs`.

        Args:
            low: The lowest possible sampling value
                (excluding this value).
            high: The highest possible sampling value
                (excluding this value).
        """
        super().__init__(inputs, model)
        # Split inputs into mean and log(std).
        mean, log_std = torch.chunk(self.inputs, 2, dim=-1)
        # Clip `scale` values (coming from NN) to reasonable values.
        log_std = torch.clamp(log_std, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT)
        std = torch.exp(log_std)
        self.base_dist = torch.distributions.normal.Normal(mean, std)
        self.dist=torch.distributions.TransformedDistribution(
            torch.distributions.Independent(self.base_dist, len(mean.shape)-1),
            torch.distributions.TanhTransform
        )

        self.mean = mean
        self.std = std

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        self.last_sample = torch.tanh(self.mean)
        return self.last_sample

    @override(TorchDistributionWrapper)
    def sample(self) -> TensorType:
        # Use the reparameterization version of `dist.sample` to allow for
        # the results to be backprop'able e.g. in a loss term.

        self.last_sample = self.dist.rsample()
        return self.last_sample

    @override(ActionDistribution)
    def logp(self, x: TensorType) -> TensorType:
        
        return self.dist.log_prob(x)

    def sample_logp(self):
        actions = self.dist.rsample()
        log_prob = self.dist.log_prob(actions)

        return actions, log_prob

    @override(TorchDistributionWrapper)
    def entropy(self) -> TensorType:
        raise ValueError("Entropy not defined for SquashedGaussian!")

    @override(TorchDistributionWrapper)
    def kl(self, other: ActionDistribution) -> TensorType:
        raise ValueError("KL not defined for SquashedGaussian!")

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(
        action_space: gym.Space, model_config: ModelConfigDict
    ) -> Union[int, np.ndarray]:
        return np.prod(action_space.shape, dtype=np.int32) * 2