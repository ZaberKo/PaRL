
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import huber_loss
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS

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


def actor_critic_loss_old(
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

    # Discrete case.
    if model.discrete:
        # Get all action probs directly from pi and form their logp.
        action_dist_inputs_t, _ = model.get_action_model_outputs(model_out_t)
        log_pis_t = F.log_softmax(action_dist_inputs_t, dim=-1)
        policy_t = torch.exp(log_pis_t)
        action_dist_inputs_tp1, _ = model.get_action_model_outputs(
            model_out_tp1)
        log_pis_tp1 = F.log_softmax(action_dist_inputs_tp1, -1)
        policy_tp1 = torch.exp(log_pis_tp1)
        # Q-values.
        q_t, _ = model.get_q_values(model_out_t)
        # Target Q-values.
        q_tp1, _ = target_model.get_q_values(target_model_out_tp1)
        if policy.config["twin_q"]:
            twin_q_t, _ = model.get_twin_q_values(model_out_t)
            twin_q_tp1, _ = target_model.get_twin_q_values(
                target_model_out_tp1)
            q_tp1 = torch.min(q_tp1, twin_q_tp1)
        q_tp1 -= alpha * log_pis_tp1

        # Actually selected Q-values (from the actions batch).
        one_hot = F.one_hot(
            train_batch[SampleBatch.ACTIONS].long(), num_classes=q_t.size()[-1]
        )
        q_t_selected = torch.sum(q_t * one_hot, dim=-1)
        if policy.config["twin_q"]:
            twin_q_t_selected = torch.sum(twin_q_t * one_hot, dim=-1)
        # Discrete case: "Best" means weighted by the policy (prob) outputs.
        q_tp1_best = torch.sum(torch.mul(policy_tp1, q_tp1), dim=-1)
        q_tp1_best_masked = (
            1.0 - train_batch[SampleBatch.DONES].float()) * q_tp1_best
    # Continuous actions case.
    else:
        # Sample single actions from distribution.
        action_dist_class = dist_class
        action_dist_inputs_t, _ = model.get_action_model_outputs(model_out_t)
        action_dist_t = action_dist_class(action_dist_inputs_t, model)
        policy_t = (
            action_dist_t.sample()
            if not deterministic
            else action_dist_t.deterministic_sample()
        )
        log_pis_t = torch.unsqueeze(action_dist_t.logp(policy_t), -1)
        action_dist_inputs_tp1, _ = model.get_action_model_outputs(
            model_out_tp1)
        action_dist_tp1 = action_dist_class(action_dist_inputs_tp1, model)
        policy_tp1 = (
            action_dist_tp1.sample()
            if not deterministic
            else action_dist_tp1.deterministic_sample()
        )
        log_pis_tp1 = torch.unsqueeze(action_dist_tp1.logp(policy_tp1), -1)

        # Q-values for the actually selected actions.
        q_t, _ = model.get_q_values(
            model_out_t, train_batch[SampleBatch.ACTIONS])
        if policy.config["twin_q"]:
            twin_q_t, _ = model.get_twin_q_values(
                model_out_t, train_batch[SampleBatch.ACTIONS]
            )

        # Q-values for current policy in given current state.
        q_t_det_policy, _ = model.get_q_values(model_out_t, policy_t)
        if policy.config["twin_q"]:
            twin_q_t_det_policy, _ = model.get_twin_q_values(
                model_out_t, policy_t)
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
        q_tp1 -= alpha.detach() * log_pis_tp1

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

    critic_loss = [torch.mean(train_batch[PRIO_WEIGHTS]
                              * huber_loss(base_td_error))]
    if policy.config["twin_q"]:
        critic_loss.append(
            torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(twin_td_error))
        )

    # Alpha- and actor losses.
    # Note: In the papers, alpha is used directly, here we take the log.
    # Discrete case: Multiply the action probs as weights with the original
    # loss terms (no expectations needed).
    if model.discrete:
        weighted_log_alpha_loss = policy_t.detach() * (
            -model.log_alpha * (log_pis_t + model.target_entropy).detach()
        )
        # Sum up weighted terms and mean over all batch items.
        alpha_loss = torch.mean(torch.sum(weighted_log_alpha_loss, dim=-1))
        # Actor loss.
        actor_loss = torch.mean(
            torch.sum(
                torch.mul(
                    # NOTE: No stop_grad around policy output here
                    # (compare with q_t_det_policy for continuous case).
                    policy_t,
                    alpha.detach() * log_pis_t - q_t.detach(),
                ),
                dim=-1,
            )
        )
    else:
        alpha_loss = -torch.mean(
            model.log_alpha * (log_pis_t + model.target_entropy).detach()
        )
        # Note: Do not detach q_t_det_policy here b/c is depends partly
        # on the policy vars (policy sample pushed through Q-net).
        # However, we must make sure `actor_loss` is not used to update
        # the Q-net(s)' variables.
        actor_loss = torch.mean(
            alpha.detach() * log_pis_t - q_t_det_policy)

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["q_t"] = q_t
    model.tower_stats["policy_t"] = policy_t
    model.tower_stats["log_pis_t"] = log_pis_t
    model.tower_stats["actor_loss"] = actor_loss
    model.tower_stats["critic_loss"] = critic_loss
    model.tower_stats["alpha_loss"] = alpha_loss

    # TD-error tensor in final stats
    # will be concatenated and retrieved for each individual batch item.
    model.tower_stats["td_error"] = td_error

    # Return all loss terms corresponding to our optimizers.
    return tuple([actor_loss] + critic_loss + [alpha_loss])


def actor_loss(
    policy: Policy,
    model: ModelV2,
    dist_class: Type[TorchDistributionWrapper],
    train_batch: SampleBatch
):


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

    # Look up the target model (tower) using the model tower.
    target_model = policy.target_models[model]

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
    action_dist_class = dist_class

    # ============== critic loss ================
    action_dist_inputs_tp1, _ = model.get_action_model_outputs(model_out_tp1)
    action_dist_tp1 = action_dist_class(action_dist_inputs_tp1, model)

    policy_tp1, log_pis_tp1 = action_dist_tp1.sample_logp()  # [B, act], [B]

    # log_pis_tp1 = torch.unsqueeze(action_dist_tp1.logp(policy_tp1), -1) #[B] -> [B,1]

    # Q-values for the actually selected actions.
    q_t, _ = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
    q_t_selected = torch.squeeze(q_t, dim=-1)  # [B,1] -> [B]
    if policy.config["twin_q"]:
        twin_q_t, _ = model.get_twin_q_values(
            model_out_t, train_batch[SampleBatch.ACTIONS]
        )
        twin_q_t_selected = torch.squeeze(twin_q_t, dim=-1)

    # Target q network evaluation.
    with torch.no_grad():
        q_tp1, _ = target_model.get_q_values(target_model_out_tp1, policy_tp1)
        if policy.config["twin_q"]:
            twin_q_tp1, _ = target_model.get_twin_q_values(
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
    critic_loss = [torch.mean(train_batch[PRIO_WEIGHTS]
                              * F.huber_loss(input=q_t_selected, target=q_t_selected_target, reduction='none'))]
    if policy.config["twin_q"]:
        critic_loss.append(
            torch.mean(train_batch[PRIO_WEIGHTS] *
                       F.huber_loss(input=twin_q_t_selected, target=q_t_selected_target, reduction='none'))
        )

    # ================ actor loss =================
    # Sample single actions from distribution.
    action_dist_inputs_t, _ = model.get_action_model_outputs(model_out_t)
    action_dist_t = action_dist_class(action_dist_inputs_t, model)

    policy_t, log_pis_t = action_dist_t.sample_logp()  # [B]

    # Q-values for current policy in given current state.
    q_t_det_policy, _ = model.get_q_values(model_out_t, policy_t)
    if policy.config["twin_q"]:
        twin_q_t_det_policy, _ = model.get_twin_q_values(model_out_t, policy_t)
        q_t_det_policy = torch.min(q_t_det_policy, twin_q_t_det_policy)

    q_t_det_policy = torch.squeeze(q_t_det_policy, dim=-1)

    actor_loss = torch.mean(alpha.detach() * log_pis_t - q_t_det_policy)

    # ============== alpha loss ================
    alpha_loss = -torch.mean(
        model.log_alpha * (log_pis_t + model.target_entropy).detach()
    )
    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["q_t"] = q_t
    model.tower_stats["policy_t"] = policy_t
    model.tower_stats["log_pis_t"] = log_pis_t
    model.tower_stats["actor_loss"] = actor_loss
    model.tower_stats["critic_loss"] = critic_loss
    model.tower_stats["alpha_loss"] = alpha_loss

    # TD-error tensor in final stats
    # will be concatenated and retrieved for each individual batch item.
    model.tower_stats["td_error"] = td_error

    # Return all loss terms corresponding to our optimizers.
    return tuple([actor_loss] + critic_loss + [alpha_loss])


# disable alpha tuning and disable priority replay
def actor_critic_loss_no_alpha(
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
    action_dist_class = dist_class

    # ============== critic loss ================
    action_dist_inputs_tp1, _ = model.get_action_model_outputs(model_out_tp1)
    action_dist_tp1 = action_dist_class(action_dist_inputs_tp1, model)

    policy_tp1, log_pis_tp1 = action_dist_tp1.sample_logp()  # [B, act], [B]

    # log_pis_tp1 = torch.unsqueeze(action_dist_tp1.logp(policy_tp1), -1) #[B] -> [B,1]

    # Q-values for the actually selected actions.
    q_t, _ = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
    q_t_selected = torch.squeeze(q_t, dim=-1)  # [B,1] -> [B]
    if policy.config["twin_q"]:
        twin_q_t, _ = model.get_twin_q_values(
            model_out_t, train_batch[SampleBatch.ACTIONS]
        )
        twin_q_t_selected = torch.squeeze(twin_q_t, dim=-1)

    # Target q network evaluation.
    with torch.no_grad():
        q_tp1, _ = target_model.get_q_values(target_model_out_tp1, policy_tp1)
        if policy.config["twin_q"]:
            twin_q_tp1, _ = target_model.get_twin_q_values(
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
    critic_loss = [torch.mean(train_batch[PRIO_WEIGHTS]
                              * F.huber_loss(input=q_t_selected, target=q_t_selected_target, reduction='none'))]
    if policy.config["twin_q"]:
        critic_loss.append(
            torch.mean(train_batch[PRIO_WEIGHTS] *
                       F.huber_loss(input=twin_q_t_selected, target=q_t_selected_target, reduction='none'))
        )

    # ================ actor loss =================
    # Sample single actions from distribution.
    action_dist_inputs_t, _ = model.get_action_model_outputs(model_out_t)
    action_dist_t = action_dist_class(action_dist_inputs_t, model)

    policy_t, log_pis_t = action_dist_t.sample_logp()  # [B]

    # Q-values for current policy in given current state.
    q_t_det_policy, _ = model.get_q_values(model_out_t, policy_t)
    if policy.config["twin_q"]:
        twin_q_t_det_policy, _ = model.get_twin_q_values(model_out_t, policy_t)
        q_t_det_policy = torch.min(q_t_det_policy, twin_q_t_det_policy)

    q_t_det_policy = torch.squeeze(q_t_det_policy, dim=-1)

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
