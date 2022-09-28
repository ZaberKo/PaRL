import tree
import ray

from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.algorithms.sac.sac_torch_policy import (
    build_sac_model_and_action_dist,
    _get_dist_class,
    optimizer_fn,
    stats,
    actor_critic_loss,
    action_distribution_fn,
    ComputeTDErrorMixin,
    TargetNetworkMixin
)
from ray.rllib.algorithms.sac.sac_tf_policy import (
    postprocess_trajectory,
    validate_spaces,
)
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    concat_multi_gpu_td_errors,
    huber_loss
)

from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.utils.framework import try_import_torch
from .sac_policy_mixin import SACEvolveMixin
from .action_dist import SquashedGaussian

import gym
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import (
    TensorType,
    AlgorithmConfigDict,
    LocalOptimizer,
    ModelInputDict
)
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models import ModelV2
from typing import List, Type, Union, Dict,Tuple, Optional

torch, nn = try_import_torch()
F = nn.functional

def optimizer_fn2(policy: Policy, config: AlgorithmConfigDict) -> Tuple[LocalOptimizer]:
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

    return action_dist_inputs, SquashedGaussian, []

def build_sac_model_and_action_dist_fix(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: AlgorithmConfigDict,
) -> Tuple[ModelV2, Type[TorchDistributionWrapper]]:
    model, action_dist_class=build_sac_model_and_action_dist(policy,obs_space,action_space,config)

    return model, SquashedGaussian

# disable alpha tuning and disable priority replay
def actor_critic_loss2(
    policy: Policy,
    model: ModelV2,
    dist_class: Type[TorchDistributionWrapper],
    train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for the Soft Actor Critic.

    Args:
        policy: The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[TorchDistributionWrapper]: The action distr. class.
        train_batch: The training data.

    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    # Look up the target model (tower) using the model tower.
    target_model = policy.target_models[model]

    # Should be True only for debugging purposes (e.g. test cases)!
    deterministic = policy.config["_deterministic_loss"]

    model_out_t, _ = model(
        SampleBatch(obs=train_batch[SampleBatch.CUR_OBS],
                    _is_training=True), [], None
    )

    model_out_tp1, _ = model(
        SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS],
                    _is_training=True), [], None
    )

    target_model_out_tp1, _ = target_model(
        SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS],
                    _is_training=True), [], None
    )

    alpha = torch.exp(model.log_alpha)

    # Sample single actions from distribution.
    action_dist_class = _get_dist_class(
        policy, policy.config, policy.action_space)
    action_dist_inputs_t, _ = model.get_action_model_outputs(model_out_t)
    action_dist_t = action_dist_class(action_dist_inputs_t, model)
    policy_t = (
        action_dist_t.sample()
        if not deterministic
        else action_dist_t.deterministic_sample()
    )
    log_pis_t = torch.unsqueeze(action_dist_t.logp(policy_t), -1)
    action_dist_inputs_tp1, _ = model.get_action_model_outputs(model_out_tp1)
    action_dist_tp1 = action_dist_class(action_dist_inputs_tp1, model)
    policy_tp1 = (
        action_dist_tp1.sample()
        if not deterministic
        else action_dist_tp1.deterministic_sample()
    )
    log_pis_tp1 = torch.unsqueeze(action_dist_tp1.logp(policy_tp1), -1)

    # Q-values for the actually selected actions.
    q_t, _ = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
    if policy.config["twin_q"]:
        twin_q_t, _ = model.get_twin_q_values(
            model_out_t, train_batch[SampleBatch.ACTIONS]
        )

    # Q-values for current policy in given current state.
    q_t_det_policy, _ = model.get_q_values(model_out_t, policy_t)
    if policy.config["twin_q"]:
        twin_q_t_det_policy, _ = model.get_twin_q_values(model_out_t, policy_t)
        q_t_det_policy = torch.min(q_t_det_policy, twin_q_t_det_policy)

    # Target q network evaluation.
    q_tp1, _ = target_model.get_q_values(target_model_out_tp1, policy_tp1)
    if policy.config["twin_q"]:
        twin_q_tp1, _ = target_model.get_twin_q_values(
            target_model_out_tp1, policy_tp1
        )
        # Take min over both twin-NNs.
        q_tp1 = torch.min(q_tp1, twin_q_tp1)

    q_t_selected = torch.squeeze(q_t, dim=-1)
    if policy.config["twin_q"]:
        twin_q_t_selected = torch.squeeze(twin_q_t, dim=-1)
    q_tp1 -= alpha * log_pis_tp1

    q_tp1_best = torch.squeeze(input=q_tp1, dim=-1)
    q_tp1_best_masked = (
        1.0 - train_batch[SampleBatch.DONES].float()) * q_tp1_best

    # compute RHS of bellman equation
    q_t_selected_target = (
        train_batch[SampleBatch.REWARDS]
        + (policy.config["gamma"] **
           policy.config["n_step"]) * q_tp1_best_masked
    ).detach()

    # Compute the TD-error (potentially clipped).
    base_td_error = torch.abs(q_t_selected - q_t_selected_target)
    if policy.config["twin_q"]:
        twin_td_error = torch.abs(twin_q_t_selected - q_t_selected_target)
        td_error = 0.5 * (base_td_error + twin_td_error)
    else:
        td_error = base_td_error

    # currently train_batch[PRIO_WEIGHTS] is always ones by `postprocess_nstep_and_prio()`
    critic_loss = [torch.mean(train_batch[PRIO_WEIGHTS]
                              * huber_loss(base_td_error))]
    if policy.config["twin_q"]:
        critic_loss.append(
            torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(twin_td_error))
        )

    # Alpha- and actor losses.
    # Note: In the papers, alpha is used directly, here we take the log.
    actor_loss = torch.mean(alpha.detach() * log_pis_t - q_t_det_policy)

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["q_t"] = q_t
    model.tower_stats["policy_t"] = policy_t
    model.tower_stats["log_pis_t"] = log_pis_t
    model.tower_stats["actor_loss"] = actor_loss
    model.tower_stats["critic_loss"] = critic_loss

    # TD-error tensor in final stats
    # will be concatenated and retrieved for each individual batch item.
    model.tower_stats["td_error"] = td_error

    # Return all loss terms corresponding to our optimizers.
    return tuple([actor_loss] + critic_loss)

def stats2(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    """Stats function for SAC. Returns a dict with important loss stats.

    Args:
        policy: The Policy to generate stats for.
        train_batch: The SampleBatch (already) used for training.

    Returns:
        Dict[str, TensorType]: The stats dict.
    """
    q_t = torch.stack(policy.get_tower_stats("q_t"))

    return {
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

def setup_late_mixins(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: AlgorithmConfigDict,
) -> None:
    ComputeTDErrorMixin.__init__(policy)
    TargetNetworkMixin.__init__(policy)
    SACEvolveMixin.__init__(policy)


# Build a child class of `TorchPolicy`, given the custom functions defined
# above.
SACPolicy = build_policy_class(
    name="SACTorchPolicy",
    framework="torch",
    loss_fn=actor_critic_loss,
    get_default_config=lambda: ray.rllib.algorithms.sac.sac.DEFAULT_CONFIG,
    stats_fn=stats,
    postprocess_fn=postprocess_trajectory,
    extra_grad_process_fn=apply_grad_clipping,
    optimizer_fn=optimizer_fn,
    validate_spaces=validate_spaces,
    before_loss_init=setup_late_mixins,
    make_model_and_action_dist=build_sac_model_and_action_dist_fix,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    mixins=[TargetNetworkMixin, ComputeTDErrorMixin, SACEvolveMixin],
    action_distribution_fn=action_distribution_fn_fix,
)


SACPolicy_FixedAlpha = build_policy_class(
    name="SACTorchPolicy",
    framework="torch",
    loss_fn=actor_critic_loss2,
    get_default_config=lambda: ray.rllib.algorithms.sac.sac.DEFAULT_CONFIG,
    stats_fn=stats2,
    postprocess_fn=postprocess_trajectory,
    extra_grad_process_fn=apply_grad_clipping,
    optimizer_fn=optimizer_fn2,
    validate_spaces=validate_spaces,
    before_loss_init=setup_late_mixins,
    make_model_and_action_dist=build_sac_model_and_action_dist_fix,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    mixins=[TargetNetworkMixin, ComputeTDErrorMixin, SACEvolveMixin],
    action_distribution_fn=action_distribution_fn_fix,
)
