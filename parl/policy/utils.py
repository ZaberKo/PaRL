import numpy as np
import tree
import copy
import ray

from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.algorithms.sac.sac_tf_policy import (
    # build_sac_model,
    postprocess_trajectory,
    validate_spaces,
)
from ray.rllib.models import ModelCatalog, MODEL_DEFAULTS


from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.utils.framework import try_import_torch
from parl.model.sac_model import SACTorchModel
from .sac_policy_mixin import (
    SACEvolveMixin, 
    SACLearning, 
    TargetNetworkMixin,
    TargetNetworkMixin2
)
from .policy_mixin import (
    TorchPolicyMod, 
    concat_multi_gpu_td_errors
)
from .action_dist import (
    SquashedGaussian, 
    SquashedGaussian2, 
    SquashedGaussian3
)
from .sac_loss import actor_critic_loss_fix, actor_critic_loss_no_alpha
from parl.utils import disable_grad_ctx, disable_grad

import gym
from ray.rllib.policy import Policy, TorchPolicy
from ray.rllib.policy.torch_policy import _directStepOptimizerSingleton
from ray.rllib.utils.typing import (
    TensorType,
    AlgorithmConfigDict,
    LocalOptimizer,
    ModelInputDict
)
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models import ModelV2
from typing import List, Type, Dict, Tuple, Optional

torch, nn = try_import_torch()
F = nn.functional

def record_grads(
    policy: "TorchPolicy", optimizer: LocalOptimizer, loss: TensorType
) -> Dict[str, TensorType]:
    """Applies gradient clipping to already computed grads inside `optimizer`.

    Args:
        policy: The TorchPolicy, which calculated `loss`.
        optimizer: A local torch optimizer object.
        loss: The torch loss tensor.

    Returns:
        An info dict containing the "grad_norm" key and the resulting clipped
        gradients.
    """
    grad_gnorm = 0

    for param_group in optimizer.param_groups:
        params = list(
            filter(lambda p: p.grad is not None, param_group["params"]))
        if params:
            grad_gnorm += torch.norm(torch.stack([
                torch.norm(p.grad.detach(), p=2)
                for p in params
            ]), p=2).cpu().numpy()

    if policy.actor_optim == optimizer:
        return {"actor_gnorm": grad_gnorm}
    elif policy.critic_optims[0] == optimizer:
        return {"critic_gnorm": grad_gnorm}
    elif policy.critic_optims[1] == optimizer:
        return {"twin_critic_gnorm": grad_gnorm}
    elif hasattr(policy, "alpha_optim") and policy.alpha_optim == optimizer:
        return {"alpha_gnorm": grad_gnorm}
    else:
        return {}


def apply_and_record_grad_clipping(
    policy: "TorchPolicy", optimizer: LocalOptimizer, loss: TensorType
) -> Dict[str, TensorType]:
    optim_config = policy.config["optimization"]
    clip_value = np.inf
    key = None

    if policy.actor_optim == optimizer:
        key = "actor_gnorm"
        clip_value = optim_config.get("actor_grad_clip", np.inf)
    elif policy.critic_optims[0] == optimizer:
        key = "critic_gnorm"
        clip_value = optim_config.get("critic_grad_clip", np.inf)
    elif policy.critic_optims[1] == optimizer:
        key = "twin_critic_gnorm"
        clip_value = optim_config.get("critic_grad_clip", np.inf)
    elif hasattr(policy, "alpha_optim") and policy.alpha_optim == optimizer:
        key = "alpha_gnorm"
        clip_value = optim_config.get("alpha_grad_clip", np.inf)
    if clip_value is None:
        clip_value = np.inf
    grad_gnorm = 0

    for param_group in optimizer.param_groups:
        # Make sure we only pass params with grad != None into torch
        # clip_grad_norm_. Would fail otherwise.
        params = list(
            filter(lambda p: p.grad is not None, param_group["params"]))
        if params:
            # PyTorch clips gradients inplace and returns the norm before clipping
            # We therefore need to compute grad_gnorm further down (fixes #4965)
            global_norm = nn.utils.clip_grad_norm_(params, clip_value)

            if isinstance(global_norm, torch.Tensor):
                global_norm = global_norm.cpu().numpy()

            grad_gnorm += global_norm

    if key is not None:
        return {key: grad_gnorm}
    else:
        return {}


def apply_gradients(policy, gradients) -> None:
    assert gradients == _directStepOptimizerSingleton
    # For policy gradient, update policy net one time v.s.
    # update critic net `policy_delay` time(s).
    if policy.global_step % policy.config.get("policy_delay", 1) == 0:
        policy.actor_optim.step()

    for critic_opt in policy.critic_optims:
        critic_opt.step()

    if hasattr(policy, "alpha_optim"):
        policy.alpha_optim.step()

    # Increment global step & apply ops.
    policy.global_step += 1