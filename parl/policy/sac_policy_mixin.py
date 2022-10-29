import torch
from .sac_loss import (
    calc_critic_loss,
    calc_actor_loss,
    calc_alpha_loss
)

from parl.utils import disable_grad_ctx

from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.numpy import convert_to_numpy

from ray.rllib.evaluation import SampleBatch
from ray.rllib.policy import TorchPolicy
from ray.rllib.utils.typing import (
    GradInfoDict,
    ModelWeights,
    TensorType,
)
from ray.rllib.utils.metrics import NUM_AGENT_STEPS_TRAINED
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from typing import Dict

class SACEvolveMixin:
    """
        Add two methods to TorchPolicy for NeuroEvolution
    """

    def __init__(self):
        # Note: apply this at `before_loss_init`
        self.params_shape = {}
        self.num_evolve_params = 0

        for name, param in self.model.action_model.named_parameters():
            self.params_shape[name] = param.size()
            self.num_evolve_params += param.numel()

    def get_evolution_weights(self) -> ModelWeights:
        # only need learnable weights in policy(actor)
        state_dict = dict(self.model.action_model.named_parameters())

        return convert_to_numpy(state_dict)

    def set_evolution_weights(self, weights: ModelWeights):
        state_dict = convert_to_torch_tensor(weights)

        self.model.action_model.load_state_dict(state_dict, strict=False)


class TargetNetworkMixin:
    """Mixin class adding a method for (soft) target net(s) synchronizations.

    - Adds the `update_target` method to the policy.
      Calling `update_target` updates all target Q-networks' weights from their
      respective "main" Q-metworks, based on tau (smooth, partial updating).
    """

    def __init__(self):
        # Hard initial update from Q-net(s) to target Q-net(s).
        self.update_target(tau=1.0)

    def update_target(self, tau=None):
        # Update_target_fn will be called periodically to copy Q network to
        # target Q network, using (soft) tau-synching.
        tau = tau or self.config.get("tau")

        with torch.no_grad():
            for p, p_target in zip (
                self.model.q_variables(),
                self.target_model.q_variables()
            ):
                p_target.mul_(1-tau)
                p_target.add_(tau*p)
            

    # @override(TorchPolicy)
    # def set_weights(self, weights):
    #     # Makes sure that whenever we restore weights for this policy's
    #     # model, we sync the target network (from the main model)
    #     # at the same time.
    #     TorchPolicy.set_weights(self, weights)
    #     self.update_target()

class TargetNetworkMixin2:
    """Mixin class adding a method for (soft) target net(s) synchronizations.

    - Adds the `update_target` method to the policy.
      Calling `update_target` updates all target Q-networks' weights from their
      respective "main" Q-metworks, based on tau (smooth, partial updating).
    """

    def __init__(self):
        # Hard initial update from Q-net(s) to target Q-net(s).
        self.update_target(tau=1.0)

    def update_target(self, tau=None):
        # Update_target_fn will be called periodically to copy Q network to
        # target Q network, using (soft) tau-synching.
        tau = tau or self.config.get("tau")

        with torch.no_grad():
            for p, p_target in zip (
                self.model.q_variables(),
                self.target_model.q_variables()
            ):
                p_target.mul_(1-tau)
                p_target.add_(tau*p)
            for p, p_target in zip (
                self.model.policy_variables(),
                self.target_model.policy_variables()
            ):
                p_target.mul_(1-tau)
                p_target.add_(tau*p)




def calc_grad_norm(optimizer):
    grad_gnorm = 0

    for param_group in optimizer.param_groups:
        params = list(
            filter(lambda p: p.grad is not None, param_group["params"]))
        if params:
            grad_gnorm += torch.norm(torch.stack([
                torch.norm(p.grad.detach(), p=2)
                for p in params
            ]), p=2).cpu().numpy()

    return grad_gnorm


class SACLearning:
    """
        Define new update pattern
    """

    def learn_on_batch(self, postprocessed_batch: SampleBatch) -> Dict[str, TensorType]:

        # Set Model to train mode.
        if self.model:
            self.model.train()
        # Callback handling.
        learn_stats = {}
        self.callbacks.on_learn_on_batch(
            policy=self, train_batch=postprocessed_batch, result=learn_stats
        )

        # grads, fetches = self.compute_gradients(postprocessed_batch)
        assert len(self.devices) == 1

        # If not done yet, see whether we have to zero-pad this batch.

        postprocessed_batch.set_training(True)
        self._lazy_tensor_dict(postprocessed_batch, device=self.devices[0])

        grad_info = {"allreduce_latency": 0.0}
        # calc gradient and updates
        critic_losses = calc_critic_loss(self, self.model, self.dist_class, postprocessed_batch)
        _name="critic_gnorm"
        for critic_loss, critic_optim in zip(critic_losses, self.critic_optims):
            critic_optim.zero_grad()
            critic_loss.backward()
            grad_info[_name]=calc_grad_norm(critic_optim)
            critic_optim.step()
            _name="twin_critic_gnorm"

        with disable_grad_ctx(self.model.q_variables()):
            actor_loss = calc_actor_loss(self, self.model, self.dist_class, postprocessed_batch)
            self.actor_optim.zero_grad()
            actor_loss.backward()
            grad_info["actor_gnorm"]=calc_grad_norm(self.actor_optim)
            self.actor_optim.step()

        if hasattr(self, "alpha_optim"):
            alpha_loss = calc_alpha_loss(self, self.model, self.dist_class, postprocessed_batch)
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            grad_info["alpha_gnorm"]=calc_grad_norm(self.alpha_optim)
            self.alpha_optim.step()

        # for TorchPolicyV2
        if hasattr(self, "stats_fn"):
            grad_info.update(self.stats_fn(postprocessed_batch))
        else:
            grad_info.update(self.extra_grad_info(postprocessed_batch))

        fetches = self.extra_compute_grad_fetches()
        fetches = dict(fetches, **{LEARNER_STATS_KEY: grad_info})

        if self.model:
            fetches["model"] = self.model.metrics()

        fetches.update(
            {
                "custom_metrics": learn_stats,
                NUM_AGENT_STEPS_TRAINED: postprocessed_batch.count,
            }
        )

        return fetches

 