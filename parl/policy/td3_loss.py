import torch
import torch.nn.functional as F
from functools import partial

from typing import List, Type, Union, Dict, Tuple, Optional
from ray.rllib.policy.policy import Policy

from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models import ModelV2
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS


def calc_actor_loss(
    policy: Policy,
    model: ModelV2,
    dist_class: Type[TorchDistributionWrapper],
    train_batch: SampleBatch
):
    input_dict = SampleBatch(
        obs=train_batch[SampleBatch.CUR_OBS], _is_training=True
    )
    # dummy function, return obs_flat
    model_out_t, _ = model(input_dict, [], None)

    # Policy network evaluation.
    policy_t = model.get_policy_output(model_out_t)

    # Q-values for current policy (no noise) in given current state
    q_t_det_policy = model.get_q_values(model_out_t, policy_t)  # [B,1]

    actor_loss = -torch.mean(q_t_det_policy)

    model.tower_stats["actor_loss"] = actor_loss

    return actor_loss


def calc_critic_loss(
    policy: Policy,
    model: ModelV2,
    dist_class: Type[TorchDistributionWrapper],
    train_batch: SampleBatch
):
    target_model = policy.target_models[model]

    twin_q = policy.config["twin_q"]
    gamma = policy.config["gamma"]
    n_step = policy.config["n_step"]
    use_huber = policy.config["use_huber"]
    huber_threshold = policy.config["huber_threshold"]
    l2_reg = policy.config["l2_reg"]

    if use_huber:
        loss_func = partial(F.huber_loss, delta=huber_threshold)
    else:
        loss_func = F.mse_loss

    input_dict = SampleBatch(
        obs=train_batch[SampleBatch.CUR_OBS], _is_training=True
    )
    input_dict_next = SampleBatch(
        obs=train_batch[SampleBatch.NEXT_OBS], _is_training=True
    )
    # dummy function, return obs_flat
    model_out_t, _ = model(input_dict, [], None)
    target_model_out_tp1, _ = target_model(input_dict_next, [], None)

    policy_tp1 = target_model.get_policy_output(target_model_out_tp1)

    # Action outputs.
    if policy.config["smooth_target_policy"]:
        target_noise_clip = policy.config["target_noise_clip"]
        clipped_normal_sample = torch.clamp(
            torch.normal(
                mean=torch.zeros(policy_tp1.size()),
                std=policy.config["target_noise"]
            ).to(policy_tp1.device),
            min=-target_noise_clip,
            max=target_noise_clip,
        )

        policy_tp1_smoothed = torch.clamp(
            policy_tp1 + clipped_normal_sample,
            min=policy.action_space_low_tensor,
            max=policy.action_space_high_tensor
        )
    else:
        # No smoothing, just use deterministic actions.
        policy_tp1_smoothed = policy_tp1

    # Q-net(s) evaluation.
    # Q-values for given actions & observations in given current
    q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
    q_t_selected = torch.squeeze(q_t, dim=-1)  # [B,1]->[B]

    if twin_q:
        twin_q_t = model.get_twin_q_values(
            model_out_t, train_batch[SampleBatch.ACTIONS]
        )
        twin_q_t_selected = torch.squeeze(twin_q_t, dim=-1)

    with torch.no_grad():
        # Target q-net(s) evaluation.
        q_tp1 = target_model.get_q_values(
            target_model_out_tp1, policy_tp1_smoothed)

        if twin_q:
            twin_q_tp1 = target_model.get_twin_q_values(
                target_model_out_tp1, policy_tp1_smoothed
            )
            q_tp1 = torch.min(q_tp1, twin_q_tp1)

        q_tp1_best = torch.squeeze(q_tp1, dim=-1)
        q_tp1_best_masked = (
            1.0 - train_batch[SampleBatch.DONES].float()) * q_tp1_best

        # Compute RHS of bellman equation.
        q_t_selected_target = (
            train_batch[SampleBatch.REWARDS] +
            gamma ** n_step * q_tp1_best_masked
        ).detach()  # [B]

    # Compute the error (potentially clipped).
    with torch.no_grad():
        base_td_error = torch.abs(q_t_selected - q_t_selected_target)
        if twin_q:
            twin_td_error = torch.abs(twin_q_t_selected -
                                      q_t_selected_target)
            td_error = 0.5 * (base_td_error + twin_td_error)
        else:
            td_error = base_td_error

    use_prio = False
    reduction = "none" if use_prio else "mean"
    if twin_q:
        errors = loss_func(
            input=q_t_selected,
            target=q_t_selected_target,
            reduction=reduction
        ) + loss_func(
            input=twin_q_t_selected,
            target=q_t_selected_target,
            reduction=reduction
        )
    else:
        errors = loss_func(
            input=q_t_selected,
            target=q_t_selected_target,
            reduction=reduction
        )

    if use_prio:
        critic_loss = torch.mean(train_batch[PRIO_WEIGHTS] * errors)
    else:
        critic_loss = errors

    model.tower_stats["q_t"] = q_t

    model.tower_stats["critic_loss"] = critic_loss
    model.tower_stats["td_error"] = td_error

    return critic_loss
