import numpy as np
import torch
import torch.nn as nn
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

        model=self.model
        target_model=self.target_models[self.model]

        with torch.no_grad():
            for p, p_target in zip(
                model.q_variables(),
                target_model.q_variables()
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




def clip_and_record_grad_norm(optimizer, clip_value=None):
    grad_gnorm = 0

    if clip_value is None:
        clip_value=np.inf

    for param_group in optimizer.param_groups:
        grad_gnorm+=nn.utils.clip_grad_norm_(param_group["params"], clip_value)

    return grad_gnorm


class SACLearning:
    """
        Define new update pattern
    """
    def __init__(self):
        self.global_step = 0

    def compute_grad_and_apply(self, train_batch):
        grad_info = {}
        # calc gradient and updates
        optim_config = self.config["optimization"]

        # ========= critic update ============
        critic_losses = calc_critic_loss(self, self.model, self.dist_class, train_batch)
        opt_name=["critic_gnorm","twin_critic_gnorm"]
        for critic_loss, critic_optim, _name in zip(critic_losses, self.critic_optims, opt_name):
            critic_optim.zero_grad()
            critic_loss.backward()
            grad_info[_name]=clip_and_record_grad_norm(
                critic_optim,
                clip_value= optim_config.get("critic_grad_clip", np.inf)
                )
            critic_optim.step()

        # ============ actor update ===============
        if self.global_step % self.config.get("policy_delay", 1) == 0:
            with disable_grad_ctx(self.model.q_variables()):
                actor_loss = calc_actor_loss(self, self.model, self.dist_class, train_batch)
                self.actor_optim.zero_grad()
                actor_loss.backward()
                grad_info["actor_gnorm"]=clip_and_record_grad_norm(
                    self.actor_optim,
                    clip_value= optim_config.get("actor_grad_clip", np.inf)
                    )
                self.actor_optim.step()

        # ============== alpha update ===============
        if hasattr(self, "alpha_optim"):
            alpha_loss = calc_alpha_loss(self, self.model, self.dist_class, train_batch)
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            grad_info["alpha_gnorm"]=clip_and_record_grad_norm(
                self.alpha_optim,
                clip_value = optim_config.get("alpha_grad_clip", np.inf)
                )
            self.alpha_optim.step()

        self.global_step += 1

        # for TorchPolicyV2
        if hasattr(self, "stats_fn"):
            grad_info.update(self.stats_fn(train_batch))
        else:
            grad_info.update(self.extra_grad_info(train_batch))

        return grad_info

    def learn_on_batch(self, postprocessed_batch: SampleBatch) -> Dict[str, TensorType]:

        # Set Model to train mode.
        if self.model:
            self.model.train()
        # Callback handling.
        custom_metrics = {}
        self.callbacks.on_learn_on_batch(
            policy=self, train_batch=postprocessed_batch, result=custom_metrics
        )

        # grads, fetches = self.compute_gradients(postprocessed_batch)
        assert len(self.devices) == 1

        # If not done yet, see whether we have to zero-pad this batch.

        postprocessed_batch.set_training(True)
        self._lazy_tensor_dict(postprocessed_batch, device=self.devices[0])

        grad_info = self.compute_grad_and_apply(postprocessed_batch)

        fetches = self.extra_compute_grad_fetches()
        fetches = dict(fetches, **{LEARNER_STATS_KEY: grad_info})

        if self.model:
            fetches["model"] = self.model.metrics()

        fetches.update(
            {
                "custom_metrics": custom_metrics,
                NUM_AGENT_STEPS_TRAINED: postprocessed_batch.count,
            }
        )

        return fetches

    def learn_on_loaded_batch(self: TorchPolicy, offset: int = 0, buffer_index: int = 0):
        if not self._loaded_batches[buffer_index]:
            raise ValueError(
                "Must call Policy.load_batch_into_buffer() before "
                "Policy.learn_on_loaded_batch()!"
            )

        assert len(self.devices) == 1

        # Get the correct slice of the already loaded batch to use,
        # based on offset and batch size.
        device_batch_size = self.config.get(
            "sgd_minibatch_size", self.config["train_batch_size"]
        ) // len(self.devices)

        # Set Model to train mode.
        if self.model:
            self.model.train()

        # only fetch gpu0 batch
        if device_batch_size >= sum(len(s) for s in self._loaded_batches[buffer_index]):
            device_batch = self._loaded_batches[buffer_index][0]
        else:
            device_batch = self._loaded_batches[buffer_index][0][offset: offset + device_batch_size]

        # Callback handling.
        batch_fetches = {}
        custom_metrics = {}
        self.callbacks.on_learn_on_batch(
            policy=self, train_batch=device_batch, result=custom_metrics
        )

        # Do the (maybe parallelized) gradient calculation step.
        grad_info = self.compute_grad_and_apply(device_batch)


        fetches = self.extra_compute_grad_fetches()
        fetches = dict(fetches, **{LEARNER_STATS_KEY: grad_info})

        if self.model:
            fetches["model"] = self.model.metrics()

        fetches.update(
            {
                "custom_metrics": custom_metrics,
                NUM_AGENT_STEPS_TRAINED: device_batch.count,
            }
        )

        return batch_fetches

 