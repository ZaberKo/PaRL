from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import huber_loss
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS

from parl.utils import disable_grad_ctx
from functools import partial

from typing import List, Type, Union, Dict, Tuple, Optional
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import (
    TensorType,
    AlgorithmConfigDict,
    LocalOptimizer,
    ModelInputDict
)
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models import ModelV2

torch, nn = try_import_torch()
F = nn.functional


def calc_actor_loss(
    policy: Policy,
    model: ModelV2,
    dist_class: Type[TorchDistributionWrapper],
    train_batch: SampleBatch
):
    alpha = torch.exp(model.log_alpha).detach()
    model_out_t, _ = model(
        SampleBatch(obs=train_batch[SampleBatch.CUR_OBS],
                    _is_training=True), [], None
    )

    # Sample single actions from distribution.
    action_dist_inputs_t = model.get_action_model_outputs(model_out_t)
    action_dist_t = dist_class(action_dist_inputs_t, model)

    policy_t, log_pis_t = action_dist_t.sample_logp()  # [B]

    # Q-values for current policy in given current state.
    q_t_det_policy = model.get_q_values(model_out_t, policy_t)
    if policy.config["twin_q"]:
        twin_q_t_det_policy = model.get_twin_q_values(model_out_t, policy_t)
        q_t_det_policy = torch.min(q_t_det_policy, twin_q_t_det_policy)
        # TODO: use torch.mean as softlearning

    q_t_det_policy = torch.squeeze(q_t_det_policy, dim=-1)

    actor_loss = torch.mean(alpha * log_pis_t - q_t_det_policy)

    model.tower_stats["policy_t"] = policy_t
    model.tower_stats["log_pis_t"] = log_pis_t

    model.tower_stats["actor_loss"] = actor_loss

    return actor_loss


def calc_critic_loss(
    policy: Policy,
    model: ModelV2,
    dist_class: Type[TorchDistributionWrapper],
    train_batch: SampleBatch
):
    use_huber=policy.config.get("use_huber", False)
    huber_beta = policy.config.get("huber_beta", 1.0)

    if use_huber:
        loss_func=partial(F.smooth_l1_loss, beta=huber_beta)
    else:
        loss_func=F.mse_loss

    # Look up the target model (tower) using the model tower.
    target_model = policy.target_models[model]
    alpha = torch.exp(model.log_alpha).detach()

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

    # Q-values for the actually selected actions.
    q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
    q_t_selected = torch.squeeze(q_t, dim=-1)  # [B,1] -> [B]
    if policy.config["twin_q"]:
        twin_q_t = model.get_twin_q_values(
            model_out_t, train_batch[SampleBatch.ACTIONS]
        )
        twin_q_t_selected = torch.squeeze(twin_q_t, dim=-1)

    # Target q network evaluation.
    with torch.no_grad():
        action_dist_inputs_tp1 = model.get_action_model_outputs(
            model_out_tp1)
        action_dist_tp1 = dist_class(action_dist_inputs_tp1, model)
        # [B, act], [B]
        policy_tp1, log_pis_tp1 = action_dist_tp1.sample_logp()

        q_tp1 = target_model.get_q_values(target_model_out_tp1, policy_tp1)
        if policy.config["twin_q"]:
            twin_q_tp1 = target_model.get_twin_q_values(
                target_model_out_tp1, policy_tp1
            )
            # Take min over both twin-NNs.
            q_tp1 = torch.min(q_tp1, twin_q_tp1)

        q_tp1 = torch.squeeze(q_tp1, dim=-1)  # [B,1] -> [B]
        q_tp1 -= alpha * log_pis_tp1

        q_tp1_masked = (1.0 - train_batch[SampleBatch.DONES].float()) * q_tp1

        # compute RHS of bellman equation
        q_t_selected_target = (
            train_batch[SampleBatch.REWARDS]
            + (policy.config["gamma"] **
               policy.config["n_step"]) * q_tp1_masked
        ).detach()

    # Compute the TD-error (potentially clipped).
    with torch.no_grad():
        base_td_error = torch.abs(q_t_selected - q_t_selected_target)  # [B]
        if policy.config["twin_q"]:
            twin_td_error = torch.abs(twin_q_t_selected - q_t_selected_target)
            td_error = 0.5 * (base_td_error + twin_td_error)
        else:
            td_error = base_td_error

    # currently train_batch[PRIO_WEIGHTS] is always ones by `postprocess_nstep_and_prio()`
    use_prio = False
    
    if use_prio:
        critic_losses = [torch.mean(train_batch[PRIO_WEIGHTS]
                                  * loss_func(input=q_t_selected, target=q_t_selected_target, reduction='none'))]
        if policy.config["twin_q"]:
            critic_losses.append(
                torch.mean(train_batch[PRIO_WEIGHTS] *
                           loss_func(input=twin_q_t_selected, target=q_t_selected_target, reduction='none'))
            )
    else:
        critic_losses = [loss_func(
            input=q_t_selected, target=q_t_selected_target, reduction='mean')]
        if policy.config["twin_q"]:
            critic_losses.append(
                loss_func(input=twin_q_t_selected,
                             target=q_t_selected_target, reduction='mean')
            )

    model.tower_stats["q_t"] = q_t
    # TD-error tensor in final stats
    # will be concatenated and retrieved for each individual batch item.
    model.tower_stats["td_error"] = td_error

    model.tower_stats["critic_loss"] = critic_losses[0]
    if policy.config["twin_q"]:
        model.tower_stats["twin_critic_loss"] = critic_losses[1]

    return critic_losses


def calc_alpha_loss(
    policy: Policy,
    model: ModelV2,
    dist_class: Type[TorchDistributionWrapper],
    train_batch: SampleBatch
):
    # TODO: this part is the only redundant part in loss,
    # where the log_pis_t is recalculated.
    with torch.no_grad():
        model_out_t, _ = model(
            SampleBatch(obs=train_batch[SampleBatch.CUR_OBS],
                        _is_training=True), [], None
        )

        # Sample single actions from distribution.
        action_dist_inputs_t = model.get_action_model_outputs(model_out_t)
        action_dist_t = dist_class(action_dist_inputs_t, model)

        policy_t, log_pis_t = action_dist_t.sample_logp()  # [B]

    alpha_loss = -torch.mean(
        model.log_alpha * (log_pis_t + model.target_entropy).detach()
    )

    model.tower_stats["alpha_loss"] = alpha_loss

    return alpha_loss


def actor_critic_loss_fix(
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
    # force limit the log_alpha for numerical stabilization
    with torch.no_grad():
        model.log_alpha.clamp_(min=-20, max=2)

    critic_losses = calc_critic_loss(policy, model, dist_class, train_batch)

    with disable_grad_ctx(model.q_variables()):
        actor_loss = calc_actor_loss(policy, model, dist_class, train_batch)

    losses = [actor_loss] + critic_losses

    if policy.config["tune_alpha"]:
        alpha_loss = calc_alpha_loss(policy, model, dist_class, train_batch)
        losses.append(alpha_loss)

    # Return all loss terms corresponding to our optimizers.
    return tuple(losses)


# # disable alpha tuning and disable priority replay
# def actor_critic_loss_no_alpha(
#     policy: Policy,
#     model: ModelV2,
#     dist_class: Type[TorchDistributionWrapper],
#     train_batch: SampleBatch,
# ) -> Union[TensorType, List[TensorType]]:
#     """Constructs the loss for the Soft Actor Critic.

#     Args:
#         policy: The Policy to calculate the loss for.
#         model (ModelV2): The Model to calculate the loss for.
#         dist_class (Type[TorchDistributionWrapper]: The action distr. class.
#         train_batch: The training data.

#     Returns:
#         Union[TensorType, List[TensorType]]: A single loss tensor or a list
#             of loss tensors.
#     """

#     critic_losses = calc_critic_loss(policy, model, dist_class, train_batch)

#     with disable_grad_ctx(model.q_variables()):
#         actor_loss = calc_actor_loss(policy, model, dist_class, train_batch)

#     # Return all loss terms corresponding to our optimizers.
#     return tuple([actor_loss] + critic_losses)
