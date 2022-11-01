import numpy as np
import gym

from parl.utils import disable_grad_ctx
from ray.rllib.algorithms.td3 import TD3Config

from parl.model.td3_model import TD3TorchModel
from .td3_policy_mixin import TD3EvolveMixin, TargetNetworkMixin, TD3Learning
from .utils import concat_multi_gpu_td_errors
from .td3_loss import (
    calc_actor_loss,
    calc_critic_loss
)


from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_action_dist import (
    TorchDeterministic,
    TorchDistributionWrapper
)
from ray.rllib.utils.annotations import override
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models import ModelV2
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.algorithms.ddpg.utils import make_ddpg_models, validate_spaces

from typing import List, Type, Union, Dict, Tuple, Optional, Any
from ray.rllib.policy import Policy, TorchPolicy
from ray.rllib.utils.typing import (
    TensorType,
    AlgorithmConfigDict
)
from ray.rllib.evaluation import Episode
from ray.rllib.algorithms.dqn.dqn_tf_policy import (
    postprocess_nstep_and_prio
)

torch, nn = try_import_torch()
F = nn.functional


def l2_loss(x: TensorType) -> TensorType:
    return 0.5 * torch.sum(torch.pow(x, 2.0))


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


class TD3Policy(TD3Learning, TargetNetworkMixin, TD3EvolveMixin, TorchPolicyV2):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: AlgorithmConfigDict,
    ):
        config = dict(TD3Config().to_dict(), **config)


        # Validate action space for DDPG
        validate_spaces(self, observation_space, action_space)

        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        self._initialize_loss_from_dummy_batch()

        TD3Learning.__init__(self)
        TargetNetworkMixin.__init__(self)
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
        self.actor_optim = torch.optim.Adam(
            params=self.model.policy_variables(), lr=self.config["actor_lr"]
        )

        self.critic_optim = torch.optim.Adam(
            params=self.model.q_variables(), lr=self.config["critic_lr"]
        )

        # Return them in the same order as the respective loss terms are returned.
        return [self.actor_optim, self.critic_optim]


    @override(TorchPolicyV2)
    def action_distribution_fn(
        self,
        model: ModelV2,
        *,
        obs_batch: TensorType,
        state_batches: TensorType,
        is_training: bool = False,
        **kwargs
    ) -> Tuple[TensorType, type, List[TensorType]]:
        model_out, _ = model(
            SampleBatch(
                obs=obs_batch[SampleBatch.CUR_OBS], _is_training=is_training)
        )
        dist_inputs = model.get_policy_output(model_out)

        distr_class = TorchDeterministic

        return dist_inputs, distr_class, []  # []=state out

    @override(TorchPolicyV2)
    def postprocess_trajectory(
        self,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[Dict[Any, SampleBatch]] = None,
        episode: Optional[Episode] = None,
    ) -> SampleBatch:
        return postprocess_nstep_and_prio(
            self, sample_batch, other_agent_batches, episode
        )

    @override(TorchPolicyV2)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch,
    ) -> List[TensorType]:
        # clip the action to avoid out of bound.
        if not hasattr(self, "action_space_low_tensor"):
            self.action_space_low_tensor = torch.from_numpy(
                self.action_space.low,
            ).to(dtype=torch.float32, device=self.device)
        if not hasattr(self, "action_space_high_tensor"):
            self.action_space_high_tensor = torch.from_numpy(
                self.action_space.high,
            ).to(dtype=torch.float32, device=self.device)

        critic_loss = calc_critic_loss(self, model, dist_class, train_batch)

        with disable_grad_ctx(model.q_variables()):
            actor_loss = calc_actor_loss(self, model, dist_class, train_batch)

        # Return two loss terms (corresponding to the two optimizers, we create).
        return [actor_loss, critic_loss]


    @override(TorchPolicyV2)
    def extra_compute_grad_fetches(self) -> Dict[str, Any]:
        fetches = convert_to_numpy(concat_multi_gpu_td_errors(self))
        return dict({LEARNER_STATS_KEY: {}}, **fetches)

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        q_t = torch.stack(self.get_tower_stats("q_t"))
        stats = {
            "actor_loss": torch.mean(torch.stack(self.get_tower_stats("actor_loss"))),
            "critic_loss": torch.mean(torch.stack(self.get_tower_stats("critic_loss"))),
            "mean_q": torch.mean(q_t),
            "max_q": torch.max(q_t),
            "min_q": torch.min(q_t),
        }
        return convert_to_numpy(stats)
