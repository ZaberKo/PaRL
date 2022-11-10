import numpy as np
import tree
import copy
import ray
import gym

from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.models import ModelCatalog, MODEL_DEFAULTS
from ray.rllib.algorithms.dqn.dqn_tf_policy import (
    postprocess_nstep_and_prio,
    PRIO_WEIGHTS,
)

from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.utils.framework import try_import_torch
from parl.model.sac_model import SACTorchModel
from .sac_policy_mixin import (
    SACEvolveMixin,
    SACLearning,
    TargetNetworkMixin
)
from .utils import concat_multi_gpu_td_errors

from .action_dist import (
    SquashedGaussian,
    SquashedGaussian2,
    SquashedGaussian3
)
from .sac_loss import actor_critic_loss_fix
from parl.utils import disable_grad


from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.typing import (
    TensorType,
    AlgorithmConfigDict,
    LocalOptimizer,
    ModelInputDict
)
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models import ModelV2
from ray.rllib.policy import Policy
from gym.spaces import Box, Discrete
from ray.rllib.utils.spaces.simplex import Simplex
from typing import List, Type, Dict, Tuple, Optional

torch, nn = try_import_torch()
F = nn.functional


def optimizer_fn(policy: Policy, config: AlgorithmConfigDict) -> Tuple[LocalOptimizer]:
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
        lr=config["optimization"]["actor_learning_rate"]
    )

    q_params=policy.model.q_variables()
    critic_split = len(q_params)
    if config["twin_q"]:
        critic_split //= 2

    policy.critic_optims = [
        torch.optim.Adam(
            params=q_params[:critic_split],
            lr=config["optimization"]["critic_learning_rate"]
        )
    ]
    if config["twin_q"]:
        policy.critic_optims.append(
            torch.optim.Adam(
                params=q_params[critic_split:],
                lr=config["optimization"]["critic_learning_rate"]
            )
        )

    optimizers = [policy.actor_optim]+policy.critic_optims

    if config["tune_alpha"]:
        policy.alpha_optim = torch.optim.Adam(
            params=[policy.model.log_alpha],
            lr=config["optimization"]["entropy_learning_rate"]
        )
        optimizers.append(policy.alpha_optim)

    return tuple(optimizers)


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
    action_dist_inputs = model.get_action_model_outputs(model_out)
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

    num_outputs = int(np.product(obs_space.shape))

    model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
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

    if config.get("is_pop_worker", False):
        disable_grad(model.parameters())
    elif not config["tune_alpha"]:
        disable_grad([model.log_alpha])

    assert isinstance(model, default_model_cls)

    # Create an exact copy of the model and store it in `policy.target_model`.
    # This will be used for tau-synched Q-target models that run behind the
    # actual Q-networks and are used for target q-value calculations in the
    # loss terms.
    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
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

    disable_grad(policy.target_model.parameters())

    assert isinstance(policy.target_model, default_model_cls)

    action_dist_class = SquashedGaussian2

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
        states["alpha_loss"] = torch.mean(
            torch.stack(policy.get_tower_stats("alpha_loss")))

    return states

def validate_spaces(
    policy: Policy,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: AlgorithmConfigDict,
) -> None:
    # Only support single Box or single Discrete spaces.
    if not isinstance(action_space, (Box, Discrete, Simplex)):
        raise UnsupportedSpaceException(
            "Action space ({}) of {} is not supported for "
            "SAC. Must be [Box|Discrete|Simplex].".format(action_space, policy)
        )
    # If Box, make sure it's a 1D vector space.
    elif isinstance(action_space, (Box, Simplex)) and len(action_space.shape) > 1:
        raise UnsupportedSpaceException(
            "Action space ({}) of {} has multiple dimensions "
            "{}. ".format(action_space, policy, action_space.shape)
            + "Consider reshaping this into a single dimension, "
            "using a Tuple action space, or the multi-agent API."
        )

def postprocess_trajectory(
    policy: Policy,
    sample_batch: SampleBatch,
    *args
) -> SampleBatch:
    return postprocess_nstep_and_prio(policy, sample_batch)

def setup_late_mixins(
    policy: Policy,
    *args
) -> None:
    SACLearning.__init__(policy)
    TargetNetworkMixin.__init__(policy)
    SACEvolveMixin.__init__(policy)


# SACPolicy = build_policy_class(
#     name="SACTorchPolicy",
#     framework="torch",
#     loss_fn=actor_critic_loss_fix,
#     get_default_config=lambda: ray.rllib.algorithms.sac.sac.DEFAULT_CONFIG,
#     stats_fn=stats,
#     postprocess_fn=postprocess_trajectory,
#     extra_grad_process_fn=apply_and_record_grad_clipping,
#     optimizer_fn=optimizer_fn,
#     validate_spaces=validate_spaces,
#     before_loss_init=setup_late_mixins,
#     make_model_and_action_dist=build_sac_model_and_action_dist_fix,
#     extra_learn_fetches_fn=concat_multi_gpu_td_errors,
#     mixins=[TorchPolicyMod, TargetNetworkMixin, SACEvolveMixin],
#     action_distribution_fn=action_distribution_fn_fix,
#     apply_gradients_fn=apply_gradients
# )


# SACPolicy_FixedAlpha = build_policy_class(
#     name="SACTorchPolicy",
#     framework="torch",
#     loss_fn=actor_critic_loss_no_alpha,
#     get_default_config=lambda: ray.rllib.algorithms.sac.sac.DEFAULT_CONFIG,
#     stats_fn=stats,
#     postprocess_fn=postprocess_trajectory,
#     # extra_grad_process_fn=apply_grad_clipping,
#     extra_grad_process_fn=apply_and_record_grad_clipping,
#     optimizer_fn=optimizer_fn_no_alpha,
#     validate_spaces=validate_spaces,
#     before_loss_init=setup_late_mixins,
#     make_model_and_action_dist=build_sac_model_and_action_dist_fix,
#     extra_learn_fetches_fn=concat_multi_gpu_td_errors,
#     mixins=[TorchPolicyMod, TargetNetworkMixin, SACEvolveMixin],
#     action_distribution_fn=action_distribution_fn_fix,
#     apply_gradients_fn=apply_gradients
# )


# SACPolicyTest = build_policy_class(
#     name="SACTorchPolicy",
#     framework="torch",
#     loss_fn=actor_critic_loss_fix,
#     get_default_config=lambda: ray.rllib.algorithms.sac.sac.DEFAULT_CONFIG,
#     stats_fn=stats,
#     postprocess_fn=postprocess_trajectory,
#     optimizer_fn=optimizer_fn,
#     validate_spaces=validate_spaces,
#     before_loss_init=setup_late_mixins,
#     make_model_and_action_dist=build_sac_model_and_action_dist_fix,
#     extra_learn_fetches_fn=concat_multi_gpu_td_errors,
#     mixins=[SACLearning, TargetNetworkMixin, SACEvolveMixin],
#     action_distribution_fn=action_distribution_fn_fix,
#     apply_gradients_fn=apply_gradients
# )


SACPolicy = build_policy_class(
    name="SACPolicy",
    framework="torch",
    loss_fn=actor_critic_loss_fix, # only use for view_req
    get_default_config=lambda: ray.rllib.algorithms.sac.sac.DEFAULT_CONFIG,
    stats_fn=stats,
    postprocess_fn=postprocess_trajectory,
    optimizer_fn=optimizer_fn,
    validate_spaces=validate_spaces,
    before_loss_init=setup_late_mixins,
    make_model_and_action_dist=build_sac_model_and_action_dist_fix,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    mixins=[SACLearning, TargetNetworkMixin, SACEvolveMixin],
    action_distribution_fn=action_distribution_fn_fix,
)
