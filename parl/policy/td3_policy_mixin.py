import numpy as np
import torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.numpy import convert_to_numpy

from parl.utils import disable_grad_ctx
from .policy_mixin import TorchPolicyCustomUpdate
from .utils import clip_and_record_grad_norm
from .td3_loss import (
    calc_actor_loss,
    calc_critic_loss
)

from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.typing import ModelWeights

from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer
from ray.rllib.policy.sample_batch import SampleBatch, concat_samples


class TD3EvolveMixin:
    """
        Add two methods to TorchPolicy for NeuroEvolution
    """

    def __init__(self):
        # Note: apply this at `before_loss_init`
        self.params_shape = {}
        self.num_evolve_params = 0

        for name, param in self.model.action_model.named_parameters():
            if self._filter(name, param):
                self.params_shape[name] = param.size()
                self.num_evolve_params += param.numel()

    def _filter(self, name, param):
        if "layer_norm" in name:
            return False
        
        return True

    def _get_evolution_weights(self) -> ModelWeights:
        # only need learnable weights in policy(actor)
        state_dict = dict(self.model.action_model.named_parameters())
        new_state_dict = {}
        for name, param in state_dict.items():
            if self._filter(name,param):
                new_state_dict[name] = param

        return new_state_dict

    def get_evolution_weights(self) -> ModelWeights:
        return convert_to_numpy(self._get_evolution_weights())

    def set_evolution_weights(self, weights: ModelWeights):
        state_dict = convert_to_torch_tensor(weights)

        self.model.action_model.load_state_dict(state_dict, strict=False)


class TD3EvolveMixinWithSM(TD3EvolveMixin):
    def __init__(self, replay_buffer_capacity=1000):
        super().__init__()
        # self.gene_replay_buffer = ReplayBuffer(
        #     capacity=replay_buffer_capacity, storage_unit="timesteps")
        self.last_samples = []

    def add_to_gene_replay_buffer(self, batches):        
        # for batch in batches:
        #     self.gene_replay_buffer.add(batch)

        self.last_samples = batches

    def calc_sensitivity(self, mutation_batch_size):
        self.model.train()

        # batch = self.gene_replay_buffer.sample(num_items=mutation_batch_size)
        batch = concat_samples(self.last_samples).shuffle()[:mutation_batch_size]

        device =self.devices[0]

        batch.set_training(True)
        self._lazy_tensor_dict(batch, device=device)

        input_dict = SampleBatch(
            obs=batch[SampleBatch.CUR_OBS], _is_training=True
        )
        # dummy function, return obs_flat
        model_out_t, _ = self.model(input_dict, [], None)

        policy_t = self.model.get_policy_output(model_out_t)

        num_actions = policy_t.shape[1]

        jacobian = torch.zeros(num_actions, self.num_evolve_params).to(device)
        grad_output = torch.zeros_like(policy_t).to(device)

        for i in range(num_actions):
            self.model.zero_grad()
            grad_output.zero_()
            grad_output[:,i]=1.0

            policy_t.backward(grad_output, retain_graph=True)
            jacobian[i]=self._get_evolve_param_flatten_grad()

        sensitivity = torch.sqrt(torch.pow(jacobian,2).sum(0))
        # sensitivity[sensitivity==0] = 1.0
        sensitivity[sensitivity<0.01] = 0.01

        return sensitivity.numpy()

    def _get_evolve_param_flatten_grad(self):
        state_dict = dict(self.model.action_model.named_parameters())

        device =self.devices[0]
        grad = torch.zeros(self.num_evolve_params,dtype=torch.float32).to(device)

        pos = 0
        for name, param in state_dict.items():
            if self._filter(name,param):
                assert param.grad is not None
                param_size = param.numel()
                grad[pos:pos+param_size] = param.grad.flatten()
                pos +=param_size

        return grad
        


class TargetNetworkMixin:
    """Mixin class adding a method for (soft) target net(s) synchronizations.

    - Adds the `update_target` method to the policy.
      Calling `update_target` updates all target Q-networks' weights from their
      respective "main" Q-metworks, based on tau (smooth, partial updating).
    """

    def __init__(self):
        # Hard initial update from Q-net(s) to target Q-net(s).
        self.update_target(tau=1.0)

    def update_target(self: TorchPolicyV2, tau=None):
        # Update_target_fn will be called periodically to copy Q network to
        # target Q network, using (soft) tau-synching.
        tau = tau or self.config.get("tau")

        model = self.model
        target_model = self.target_models[self.model]

        with torch.no_grad():
            for p, p_target in zip(
                model.parameters(),
                target_model.parameters()
            ):
                p_target.mul_(1-tau)
                p_target.add_(tau*p)

    # @override(TorchPolicyV2)
    # def set_weights(self: TorchPolicyV2, weights):
    #     # Makes sure that whenever we restore weights for this policy's
    #     # model, we sync the target network (from the main model)
    #     # at the same time.
    #     TorchPolicyV2.set_weights(self, weights)
    #     self.update_target()


class TD3Learning(TorchPolicyCustomUpdate):
    def __init__(self):
        self.global_trainstep = 0

        self.last_actor_gnorm = None

    def _compute_grad_and_apply(self, train_batch):
        grad_info = {}
        # calc gradient and updates
        # optim_config = self.config["optimization"]

        # ========= critic update ============
        critic_loss = calc_critic_loss(
            self, self.model, self.dist_class, train_batch)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        grad_info["critic_gnorm"] = clip_and_record_grad_norm(
            self.critic_optim
        )
        self.critic_optim.step()

        # ============ actor update ===============
        if self.global_trainstep % self.config.get("policy_delay", 1) == 0:
            with disable_grad_ctx(self.model.q_variables()):
                actor_loss = calc_actor_loss(
                    self, self.model, self.dist_class, train_batch)
                self.actor_optim.zero_grad()
                actor_loss.backward()
                grad_info["actor_gnorm"] = clip_and_record_grad_norm(
                    self.actor_optim
                )
                self.last_actor_gnorm = grad_info["actor_gnorm"]
                self.actor_optim.step()
        else:
            grad_info["actor_gnorm"] = self.last_actor_gnorm

        self.global_trainstep += 1

        return grad_info
