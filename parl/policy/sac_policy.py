import numpy as np
import tree
import copy
import ray

from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.algorithms.sac.sac_torch_policy import (
    optimizer_fn,
    ComputeTDErrorMixin,
    TargetNetworkMixin
)
from ray.rllib.algorithms.sac.sac_tf_policy import (
    # build_sac_model,
    postprocess_trajectory,
    validate_spaces,
)

from ray.rllib.models import ModelCatalog, MODEL_DEFAULTS


from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.utils.framework import try_import_torch
from .sac_model import SACTorchModel
from .sac_policy_mixin import SACEvolveMixin, SACLearning
from .policy_mixin import TorchPolicyMod
from .action_dist import SquashedGaussian
from .sac_loss import actor_critic_loss_fix, actor_critic_loss_no_alpha

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
from typing import List, Type, Union, Dict, Tuple, Optional

torch, nn = try_import_torch()
F = nn.functional


def optimizer_fn_no_alpha(policy: Policy, config: AlgorithmConfigDict) -> Tuple[LocalOptimizer]:
    """Creates all necessary optimizers for SAC learning.

    The 3 or 4 (twin_q=True) optimizers returned here correspond to the
    number of loss terms returned by the loss function.

    Args:
        policy: The policy object to be trained.
        config: The Algorithm's config dict.

    Returns:
        Tuple[LocalOptimizer]: The local optimizers to use for policy training.
    """
    policy.actor_optim = torch.optim.Adam(
        params=policy.model.policy_variables(),
        lr=config["optimization"]["actor_learning_rate"],
        eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
    )

    critic_split = len(policy.model.q_variables())
    if config["twin_q"]:
        critic_split //= 2

    policy.critic_optims = [
        torch.optim.Adam(
            params=policy.model.q_variables()[:critic_split],
            lr=config["optimization"]["critic_learning_rate"],
            eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
        )
    ]
    if config["twin_q"]:
        policy.critic_optims.append(
            torch.optim.Adam(
                params=policy.model.q_variables()[critic_split:],
                lr=config["optimization"]["critic_learning_rate"],
                eps=1e-7,  # to match tf.keras.optimizers.Adam's eps default
            )
        )
    # policy.alpha_optim = torch.optim.Adam(
    #     params=[policy.model.log_alpha],
    #     lr=config["optimization"]["entropy_learning_rate"],
    #     eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
    # )

    return tuple([policy.actor_optim] + policy.critic_optims)


def action_distribution_fn_fix(
    policy: Policy,
    model: ModelV2,
    input_dict: ModelInputDict,
    *,
    state_batches: Optional[List[TensorType]] = None,
    seq_lens: Optional[TensorType] = None,
    prev_action_batch: Optional[TensorType] = None,
    prev_reward_batch=None,
    explore: Optional[bool] = None,
    timestep: Optional[int] = None,
    is_training: Optional[bool] = None
) -> Tuple[TensorType, Type[TorchDistributionWrapper], List[TensorType]]:
    # Get base-model output (w/o the SAC specific parts of the network).
    model_out, _ = model(input_dict, [], None)
    # Use the base output to get the policy outputs from the SAC model's
    # policy components.
    action_dist_inputs, _ = model.get_action_model_outputs(model_out)
    # Get a distribution class to be used with the just calculated dist-inputs.

    # assert issubclass(policy.dist_class, SquashedGaussian)

    return action_dist_inputs, policy.dist_class, []


def build_sac_model_and_action_dist_fix(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: AlgorithmConfigDict,
) -> Tuple[ModelV2, Type[TorchDistributionWrapper]]:
    # Force-ignore any additionally provided hidden layer sizes.
    # Everything should be configured using SAC's `q_model_config` and
    # `policy_model_config` config settings.
    policy_model_config = copy.deepcopy(MODEL_DEFAULTS)
    policy_model_config.update(config["policy_model_config"])
    q_model_config = copy.deepcopy(MODEL_DEFAULTS)
    q_model_config.update(config["q_model_config"])

    default_model_cls = SACTorchModel

    model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=None,
        model_config=config["model"],
        framework=config["framework"],
        default_model=default_model_cls,
        name="sac_model",
        policy_model_config=policy_model_config,
        q_model_config=q_model_config,
        twin_q=config["twin_q"],
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"],
    )

    assert isinstance(model, default_model_cls)

    # Create an exact copy of the model and store it in `policy.target_model`.
    # This will be used for tau-synched Q-target models that run behind the
    # actual Q-networks and are used for target q-value calculations in the
    # loss terms.
    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=None,
        model_config=config["model"],
        framework=config["framework"],
        default_model=default_model_cls,
        name="target_sac_model",
        policy_model_config=policy_model_config,
        q_model_config=q_model_config,
        twin_q=config["twin_q"],
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"],
    )

    assert isinstance(policy.target_model, default_model_cls)

    action_dist_class = SquashedGaussian

    return model, action_dist_class


def stats(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    """Stats function for SAC. Returns a dict with important loss stats.

    Args:
        policy: The Policy to generate stats for.
        train_batch: The SampleBatch (already) used for training.

    Returns:
        Dict[str, TensorType]: The stats dict.
    """
    q_t = torch.stack(policy.get_tower_stats("q_t"))

    states = {
        "actor_loss": torch.mean(torch.stack(policy.get_tower_stats("actor_loss"))),
        "critic_loss": torch.mean(
            torch.stack(tree.flatten(policy.get_tower_stats("critic_loss")))
        ),
        "alpha_value": torch.exp(policy.model.log_alpha),
        "log_alpha_value": policy.model.log_alpha,
        "target_entropy": policy.model.target_entropy,
        "policy_t": torch.mean(torch.stack(policy.get_tower_stats("policy_t"))),
        "mean_q": torch.mean(q_t),
        "max_q": torch.max(q_t),
        "min_q": torch.min(q_t),
    }

    if hasattr(policy, "alpha_optim"):
        states["alpha_loss"]= torch.mean(torch.stack(policy.get_tower_stats("alpha_loss")))

    return states



def concat_multi_gpu_td_errors(
    policy: Union["TorchPolicy", "TorchPolicyV2"]
) -> Dict[str, TensorType]:
    """Concatenates multi-GPU (per-tower) TD error tensors given TorchPolicy.

    TD-errors are extracted from the TorchPolicy via its tower_stats property.

    Args:
        policy: The TorchPolicy to extract the TD-error values from.

    Returns:
        A dict mapping strings "td_error" and "mean_td_error" to the
        corresponding concatenated and mean-reduced values.
    """
    td_error = torch.cat(
        [
            t.tower_stats.get("td_error", torch.tensor([0.0])).to(policy.device)
            for t in policy.model_gpu_towers
        ],
        dim=0,
    )
    policy.td_error = td_error
    return {
        # "td_error": td_error,
        "mean_td_error": torch.mean(td_error),
    }

def setup_late_mixins(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: AlgorithmConfigDict,
) -> None:
    # SACMod.__init__(policy)
    ComputeTDErrorMixin.__init__(policy)
    TargetNetworkMixin.__init__(policy)
    SACEvolveMixin.__init__(policy)

    # used in apply_gradients()
    policy.global_step = 0


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
    elif hasattr(policy, "alpha_optim") and policy.alpha_optim==optimizer:
        return {"alpha_gnorm": grad_gnorm}
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


# Build a child class of `TorchPolicy`, given the custom functions defined
# above.
SACPolicy = build_policy_class(
    name="SACTorchPolicy",
    framework="torch",
    loss_fn=actor_critic_loss_fix,
    get_default_config=lambda: ray.rllib.algorithms.sac.sac.DEFAULT_CONFIG,
    stats_fn=stats,
    postprocess_fn=postprocess_trajectory,
    extra_grad_process_fn=record_grads,
    optimizer_fn=optimizer_fn,
    validate_spaces=validate_spaces,
    before_loss_init=setup_late_mixins,
    make_model_and_action_dist=build_sac_model_and_action_dist_fix,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    mixins=[TorchPolicyMod, TargetNetworkMixin, ComputeTDErrorMixin, SACEvolveMixin],
    action_distribution_fn=action_distribution_fn_fix,
    apply_gradients_fn=apply_gradients
)


SACPolicy_FixedAlpha = build_policy_class(
    name="SACTorchPolicy",
    framework="torch",
    loss_fn=actor_critic_loss_no_alpha,
    get_default_config=lambda: ray.rllib.algorithms.sac.sac.DEFAULT_CONFIG,
    stats_fn=stats,
    postprocess_fn=postprocess_trajectory,
    # extra_grad_process_fn=apply_grad_clipping,
    extra_grad_process_fn=record_grads,
    optimizer_fn=optimizer_fn_no_alpha,
    validate_spaces=validate_spaces,
    before_loss_init=setup_late_mixins,
    make_model_and_action_dist=build_sac_model_and_action_dist_fix,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    mixins=[TorchPolicyMod, TargetNetworkMixin, ComputeTDErrorMixin, SACEvolveMixin],
    action_distribution_fn=action_distribution_fn_fix,
    apply_gradients_fn=apply_gradients
)


SACPolicyTest = build_policy_class(
    name="SACTorchPolicy",
    framework="torch",
    loss_fn=actor_critic_loss_fix,
    get_default_config=lambda: ray.rllib.algorithms.sac.sac.DEFAULT_CONFIG,
    stats_fn=stats,
    postprocess_fn=postprocess_trajectory,
    optimizer_fn=optimizer_fn,
    validate_spaces=validate_spaces,
    before_loss_init=setup_late_mixins,
    make_model_and_action_dist=build_sac_model_and_action_dist_fix,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    mixins=[SACLearning, TargetNetworkMixin, ComputeTDErrorMixin, SACEvolveMixin],
    action_distribution_fn=action_distribution_fn_fix,
    apply_gradients_fn=apply_gradients
)
