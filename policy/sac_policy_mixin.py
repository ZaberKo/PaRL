from ray.rllib.utils.typing import ModelWeights
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.numpy import convert_to_numpy

from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.policy.torch_policy import TorchPolicy, _directStepOptimizerSingleton

import torch
import numpy as np

from ray.rllib.utils import deep_update


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


def learn_on_loaded_batch(self: TorchPolicy, offset: int = 0, buffer_index: int = 0):
    if not self._loaded_batches[buffer_index]:
        raise ValueError(
            "Must call Policy.load_batch_into_buffer() before "
            "Policy.learn_on_loaded_batch()!"
        )

    # Get the correct slice of the already loaded batch to use,
    # based on offset and batch size.
    device_batch_size = self.config.get(
        "sgd_minibatch_size", self.config["train_batch_size"]
    ) // len(self.devices)

    # Set Model to train mode.
    if self.model_gpu_towers:
        for t in self.model_gpu_towers:
            t.train()

    # Shortcut for 1 CPU only: Batch should already be stored in
    # `self._loaded_batches`.
    if len(self.devices) == 1 and self.devices[0].type == "cpu":
        assert buffer_index == 0
        if device_batch_size >= len(self._loaded_batches[0][0]):
            batch = self._loaded_batches[0][0]
        else:
            batch = self._loaded_batches[0][0][offset: offset +
                                               device_batch_size]
        return self.learn_on_batch(batch)

    if len(self.devices) > 1:
        # Copy weights of main model (tower-0) to all other towers.
        state_dict = self.model.state_dict()
        # Just making sure tower-0 is really the same as self.model.
        assert self.model_gpu_towers[0] is self.model
        for tower in self.model_gpu_towers[1:]:
            tower.load_state_dict(state_dict)

    if device_batch_size >= sum(len(s) for s in self._loaded_batches[buffer_index]):
        device_batches = self._loaded_batches[buffer_index]
    else:
        device_batches = [
            b[offset: offset + device_batch_size]
            for b in self._loaded_batches[buffer_index]
        ]

    # Callback handling.
    batch_fetches = {}
    for i, batch in enumerate(device_batches):
        custom_metrics = {}
        self.callbacks.on_learn_on_batch(
            policy=self, train_batch=batch, result=custom_metrics
        )
        batch_fetches[f"tower_{i}"] = {"custom_metrics": custom_metrics}

    # Do the (maybe parallelized) gradient calculation step.
    tower_outputs = self._multi_gpu_parallel_grad_calc(device_batches)

    # Mean-reduce gradients over GPU-towers (do this on CPU: self.device).
    all_grads = []
    for i in range(len(tower_outputs[0][0])):
        if tower_outputs[0][0][i] is not None:
            all_grads.append(
                torch.mean(
                    torch.stack([t[0][i].to(self.device)
                                for t in tower_outputs]),
                    dim=0,
                )
            )
        else:
            all_grads.append(None)
    # Set main model's grads to mean-reduced values.
    for i, p in enumerate(self.model.parameters()):
        p.grad = all_grads[i]

    self.apply_gradients(_directStepOptimizerSingleton)

    for i, (model, batch) in enumerate(zip(self.model_gpu_towers, device_batches)):
        batch_fetches[f"tower_{i}"]=deep_update(batch_fetches[f"tower_{i}"],{
            LEARNER_STATS_KEY: self.extra_grad_info(batch),
            "model": model.metrics(),
        },True)

    batch_fetches.update(self.extra_compute_grad_fetches())

    return batch_fetches
