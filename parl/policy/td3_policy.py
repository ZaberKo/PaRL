import numpy as np
import gym
from ray.rllib.algorithms.ddpg.ddpg_torch_policy import DDPGTorchPolicy

from parl.model.td3_model import TD3TorchModel
from .td3_policy_mixin import TD3EvolveMixin

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_action_dist import (
    TorchDeterministic,
    TorchDirichlet,
    TorchDistributionWrapper,
)
from ray.rllib.utils.annotations import override
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models import ModelV2
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy

from ray.rllib.utils.framework import try_import_torch

from typing import List, Type, Union, Dict, Tuple, Optional, Any
from ray.rllib.policy import Policy, TorchPolicy
from ray.rllib.utils.typing import (
    ModelGradients,
    TensorType,
    AlgorithmConfigDict,
    LocalOptimizer
)
from ray.rllib.algorithms.dqn.dqn_tf_policy import (
    PRIO_WEIGHTS,
)

torch, nn = try_import_torch()
F = nn.functional


def l2_loss(x: TensorType) -> TensorType:
    return 0.5 * torch.sum(torch.pow(x, 2.0))


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
            t.tower_stats.get("td_error", torch.tensor(
                [0.0])).to(policy.device)
            for t in policy.model_gpu_towers
        ],
        dim=0,
    )
    policy.td_error = td_error
    return {
        # "td_error": td_error,
        "mean_td_error": torch.mean(td_error),
    }

def make_ddpg_models(policy: Policy) -> ModelV2:
    num_outputs = int(np.product(policy.observation_space.shape))
    model = ModelCatalog.get_model_v2(
        obs_space=policy.observation_space,
        action_space=policy.action_space,
        num_outputs=num_outputs,
        model_config=policy.config["model"],
        framework=policy.config["framework"],
        default_model=TD3TorchModel,
        name="td3_model",
        actor_hidden_activation=policy.config["actor_hidden_activation"],
        actor_hiddens=policy.config["actor_hiddens"],
        critic_hidden_activation=policy.config["critic_hidden_activation"],
        critic_hiddens=policy.config["critic_hiddens"],
        twin_q=policy.config["twin_q"],
        add_layer_norm=(
            policy.config["exploration_config"].get("type") == "ParameterNoise"
        ),
    )

    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=policy.observation_space,
        action_space=policy.action_space,
        num_outputs=num_outputs,
        model_config=policy.config["model"],
        framework=policy.config["framework"],
        default_model=TD3TorchModel,
        name="target_td3_model",
        actor_hidden_activation=policy.config["actor_hidden_activation"],
        actor_hiddens=policy.config["actor_hiddens"],
        critic_hidden_activation=policy.config["critic_hidden_activation"],
        critic_hiddens=policy.config["critic_hiddens"],
        twin_q=policy.config["twin_q"],
        add_layer_norm=(
            policy.config["exploration_config"].get("type") == "ParameterNoise"
        ),
    )

    return model


class TD3Policy(DDPGTorchPolicy, TD3EvolveMixin):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: AlgorithmConfigDict,
    ):
        # Note: self.loss() is called in it.
        DDPGTorchPolicy.__init__(self, observation_space, action_space, config)
        TD3EvolveMixin.__init__(self)

    @override(TorchPolicyV2)
    def make_model_and_action_dist(
        self,
    ) -> Tuple[ModelV2, Type[TorchDistributionWrapper]]:
        model = make_ddpg_models(self)

        distr_class = TorchDeterministic
        return model, distr_class

    @override(TorchPolicyV2)
    def optimizer(
        self,
    ) -> List["torch.optim.Optimizer"]:
        """Create separate optimizers for actor & critic losses."""

        # Set epsilons to match tf.keras.optimizers.Adam's epsilon default.
        self.actor_optimizer = torch.optim.Adam(
            params=self.model.policy_variables(), lr=self.config["actor_lr"], eps=1e-7
        )

        self.critic_optimizer = torch.optim.Adam(
            params=self.model.q_variables(), lr=self.config["critic_lr"], eps=1e-7
        )

        # Return them in the same order as the respective loss terms are returned.
        return [self.actor_optimizer, self.critic_optimizer]

    @override(TorchPolicyV2)
    def apply_gradients(self, gradients: ModelGradients) -> None:
        # For policy gradient, update policy net one time v.s.
        # update critic net `policy_delay` time(s).
        if self.global_step % self.config["policy_delay"] == 0:
            self.actor_optimizer.step()

        self.critic_optimizer.step()

        # Increment global step & apply ops.
        self.global_step += 1

    @override(TorchPolicyV2)
    def extra_grad_process(
        self, optimizer: torch.optim.Optimizer, loss: TensorType
    ) -> Dict[str, TensorType]:
        # Clip grads if configured.
        grad_gnorm = 0

        for param_group in optimizer.param_groups:
            params = list(
                filter(lambda p: p.grad is not None, param_group["params"]))
            if params:
                grad_gnorm += torch.norm(torch.stack([
                    torch.norm(p.grad.detach(), p=2)
                    for p in params
                ]), p=2).cpu().numpy()

        if self.actor_optimizer == optimizer:
            return {"actor_gnorm": grad_gnorm}
        elif self.critic_optimizer == optimizer:
            return {"critic_gnorm": grad_gnorm}
        else:
            return {}

    @override(TorchPolicyV2)
    def extra_compute_grad_fetches(self) -> Dict[str, Any]:
        fetches = convert_to_numpy(concat_multi_gpu_td_errors(self))
        return dict({LEARNER_STATS_KEY: {}}, **fetches)

    @override(TorchPolicyV2)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch,
    ) -> List[TensorType]:
        target_model = self.target_models[model]

        twin_q = self.config["twin_q"]
        gamma = self.config["gamma"]
        n_step = self.config["n_step"]
        use_huber = self.config["use_huber"]
        huber_threshold = self.config["huber_threshold"]
        l2_reg = self.config["l2_reg"]

        # clip the action to avoid out of bound.
        if not hasattr(self, "action_space_low_tensor"):
            self.action_space_low_tensor = torch.from_numpy(
                self.action_space.low,
            ).to(dtype=torch.float32, device=self.device)
        if not hasattr(self, "action_space_high_tensor"):
            self.action_space_high_tensor = torch.from_numpy(
                self.action_space.high.copy(),
            ).to(dtype=torch.float32, device=self.device)

        input_dict = SampleBatch(
            obs=train_batch[SampleBatch.CUR_OBS], _is_training=True
        )
        input_dict_next = SampleBatch(
            obs=train_batch[SampleBatch.NEXT_OBS], _is_training=True
        )

        # ============= actor loss =================
        # dummy function, return obs_flat
        model_out_t, _ = model(input_dict, [], None)

        # Policy network evaluation.
        policy_t = model.get_policy_output(model_out_t)

        # Q-values for current policy (no noise) in given current state
        q_t_det_policy = model.get_q_values(model_out_t, policy_t)  # [B,1]

        actor_loss = -torch.mean(q_t_det_policy)

        # =========== critic loss ===============
        target_model_out_tp1, _ = target_model(input_dict_next, [], None)
        policy_tp1 = target_model.get_policy_output(target_model_out_tp1)

        # Action outputs.
        if self.config["smooth_target_policy"]:
            target_noise_clip = self.config["target_noise_clip"]
            clipped_normal_sample = torch.clamp(
                torch.normal(
                    mean=torch.zeros(policy_tp1.size()),
                    std=self.config["target_noise"]
                ).to(policy_tp1.device),
                min=-target_noise_clip,
                max=target_noise_clip,
            )

            policy_tp1_smoothed = torch.clamp(
                policy_tp1 + clipped_normal_sample,
                min=self.action_space_low_tensor,
                max=self.action_space_high_tensor
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
        use_prio = False
        reduction = "none" if use_prio else "mean"
        if twin_q:
            td_error = q_t_selected - q_t_selected_target
            if use_huber:
                errors = F.huber_loss(
                    input=q_t_selected,
                    target=q_t_selected_target,
                    delta=huber_threshold, reduction=reduction
                ) + F.huber_loss(
                    input=twin_q_t_selected,
                    target=q_t_selected_target,
                    delta=huber_threshold, reduction=reduction
                )
            else:
                errors = F.mse_loss(
                    input=q_t_selected,
                    target=q_t_selected_target,
                    reduction=reduction
                ) + F.mse_loss(
                    input=twin_q_t_selected,
                    target=q_t_selected_target,
                    reduction=reduction
                )

        else:
            td_error = q_t_selected - q_t_selected_target
            if use_huber:
                errors = F.huber_loss(
                    input=q_t_selected,
                    target=q_t_selected_target,
                    delta=huber_threshold, reduction=reduction
                )
            else:
                errors = F.mse_loss(
                    input=q_t_selected,
                    target=q_t_selected_target,
                    reduction=reduction
                )
        if use_prio:
            critic_loss = torch.mean(train_batch[PRIO_WEIGHTS] * errors)
        else:
            critic_loss = errors

        # Add l2-regularization if required.
        # if l2_reg is not None:
        #     for name, var in model.policy_variables(as_dict=True).items():
        #         if "bias" not in name:
        #             actor_loss += l2_reg * l2_loss(var)
        #     for name, var in model.q_variables(as_dict=True).items():
        #         if "bias" not in name:
        #             critic_loss += l2_reg * l2_loss(var)
        # TODO: add l2_loss at optimizer param_group level (more efficient)

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["q_t"] = q_t
        model.tower_stats["actor_loss"] = actor_loss
        model.tower_stats["critic_loss"] = critic_loss
        # TD-error tensor in final stats
        # will be concatenated and retrieved for each individual batch item.
        model.tower_stats["td_error"] = td_error

        # Return two loss terms (corresponding to the two optimizers, we create).
        return [actor_loss, critic_loss]
