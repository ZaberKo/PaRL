import numpy as np
import torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.numpy import convert_to_numpy

from parl.utils import disable_grad_ctx
from .policy_mixin import TorchPolicyCustomUpdate
from .utils import clip_and_record_grad_norm
from .td3_loss import (
    calc_actor_loss,
    calc_critic_loss
)

from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.typing import ModelWeights


class TD3EvolveMixin:
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

    def _filter(self, state_dict: ModelWeights):
        new_state_dict = {}
        for name, param in state_dict.items():
            if "layer_norm" in name:
                continue
            new_state_dict[name] = param

        return new_state_dict

    def get_evolution_weights(self) -> ModelWeights:
        # only need learnable weights in policy(actor)
        state_dict = dict(self.model.action_model.named_parameters())

        return convert_to_numpy(self._filter(state_dict))

    def set_evolution_weights(self, weights: ModelWeights):
        state_dict = convert_to_torch_tensor(weights)

        self.model.action_model.load_state_dict(state_dict, strict=False)


class TargetNetworkMixin:
    """Mixin class adding a method for (soft) target net(s) synchronizations.

    - Adds the `update_target` method to the policy.
      Calling `update_target` updates all target Q-networks' weights from their
      respective "main" Q-metworks, based on tau (smooth, partial updating).
    """

    def __init__(self):
        # Hard initial update from Q-net(s) to target Q-net(s).
        self.update_target(tau=1.0)

    def update_target(self: TorchPolicyV2, tau=None):
        # Update_target_fn will be called periodically to copy Q network to
        # target Q network, using (soft) tau-synching.
        tau = tau or self.config.get("tau")

        model = self.model
        target_model = self.target_models[self.model]

        with torch.no_grad():
            for p, p_target in zip(
                model.parameters(),
                target_model.parameters()
            ):
                p_target.mul_(1-tau)
                p_target.add_(tau*p)

    # @override(TorchPolicyV2)
    # def set_weights(self: TorchPolicyV2, weights):
    #     # Makes sure that whenever we restore weights for this policy's
    #     # model, we sync the target network (from the main model)
    #     # at the same time.
    #     TorchPolicyV2.set_weights(self, weights)
    #     self.update_target()


class TD3Learning(TorchPolicyCustomUpdate):
    def __init__(self):
        self.global_trainstep = 0

        self.last_actor_gnorm = None

    def _compute_grad_and_apply(self, train_batch):
        grad_info = {}
        # calc gradient and updates
        # optim_config = self.config["optimization"]

        # ========= critic update ============
        critic_loss = calc_critic_loss(
            self, self.model, self.dist_class, train_batch)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        grad_info["critic_gnorm"] = clip_and_record_grad_norm(
            self.critic_optim
        )
        self.critic_optim.step()

        # ============ actor update ===============
        if self.global_trainstep % self.config.get("policy_delay", 1) == 0:
            with disable_grad_ctx(self.model.q_variables()):
                actor_loss = calc_actor_loss(
                    self, self.model, self.dist_class, train_batch)
                self.actor_optim.zero_grad()
                actor_loss.backward()
                grad_info["actor_gnorm"] = clip_and_record_grad_norm(
                    self.actor_optim
                )
                self.last_actor_gnorm = grad_info["actor_gnorm"]
                self.actor_optim.step()
        else:
            grad_info["actor_gnorm"] = self.last_actor_gnorm

        self.global_trainstep += 1

        return grad_info
