import numpy as np
import torch
import torch.nn as nn
from .sac_loss import (
    calc_critic_loss,
    calc_actor_loss,
    calc_alpha_loss
)

from parl.utils import disable_grad_ctx
from .policy_mixin import TorchPolicyCustomUpdate
from .utils import clip_and_record_grad_norm

from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.numpy import convert_to_numpy


from ray.rllib.utils.typing import (
    GradInfoDict,
    ModelWeights,
    TensorType,
)

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


class TargetNetworkMixin:
    """Mixin class adding a method for (soft) target net(s) synchronizations.

    - Adds the `update_target` method to the policy.
      Calling `update_target` updates all target Q-networks' weights from their
      respective "main" Q-metworks, based on tau (smooth, partial updating).
    """

    def __init__(self):
        # Hard initial update from Q-net(s) to target Q-net(s).
        self.update_target(tau=1.0)

    def update_target(self, tau=None):
        # Update_target_fn will be called periodically to copy Q network to
        # target Q network, using (soft) tau-synching.
        tau = tau or self.config.get("tau")

        model=self.model
        target_model=self.target_models[self.model]

        with torch.no_grad():
            for p, p_target in zip(
                model.q_variables(),
                target_model.q_variables()
            ):
                p_target.mul_(1-tau)
                p_target.add_(tau*p)
            
            
class SACLearning(TorchPolicyCustomUpdate):
    def __init__(self):
        self.global_trainstep = 0

    def _compute_grad_and_apply(self, train_batch):
        grad_info = {}
        # calc gradient and updates
        optim_config = self.config["optimization"]

        # ========= critic update ============
        critic_losses = calc_critic_loss(self, self.model, self.dist_class, train_batch)
        opt_name=["critic_gnorm","twin_critic_gnorm"]
        for critic_loss, critic_optim, _name in zip(critic_losses, self.critic_optims, opt_name):
            critic_optim.zero_grad()
            critic_loss.backward()
            grad_info[_name]=clip_and_record_grad_norm(
                critic_optim,
                clip_value= optim_config.get("critic_grad_clip", np.inf)
                )
            critic_optim.step()

        # ============ actor update ===============
        if self.global_trainstep % self.config.get("policy_delay", 1) == 0:
            with disable_grad_ctx(self.model.q_variables()):
                actor_loss = calc_actor_loss(self, self.model, self.dist_class, train_batch)
                self.actor_optim.zero_grad()
                actor_loss.backward()
                grad_info["actor_gnorm"]=clip_and_record_grad_norm(
                    self.actor_optim,
                    clip_value= optim_config.get("actor_grad_clip", np.inf)
                    )
                self.actor_optim.step()

        # ============== alpha update ===============
        if hasattr(self, "alpha_optim"):
            alpha_loss = calc_alpha_loss(self, self.model, self.dist_class, train_batch)
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            grad_info["alpha_gnorm"]=clip_and_record_grad_norm(
                self.alpha_optim,
                clip_value = optim_config.get("alpha_grad_clip", np.inf)
                )
            self.alpha_optim.step()

        self.global_trainstep += 1

        return grad_info

 