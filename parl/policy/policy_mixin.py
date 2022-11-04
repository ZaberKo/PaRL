import time
import threading
import torch

from ray.rllib.utils import NullContextManager, force_list

from typing import List, Union, Dict, Tuple
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.policy.torch_policy import TorchPolicy, _directStepOptimizerSingleton
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.evaluation import SampleBatch
from ray.rllib.utils.typing import (
    GradInfoDict,
    ModelWeights,
    TensorType,
)
from ray.rllib.utils.metrics import NUM_AGENT_STEPS_TRAINED
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY



class TorchPolicyCustomUpdate:
    """
        Customed update steps: override TorchPolicy's learn_on_batch() and learn_on_loaded_batch()
        Note: Currently only support one GPU.
    """
    def _compute_grad_and_apply(self, train_batch)->GradInfoDict:
        raise NotImplementedError
    
    def compute_grad_and_apply(self, train_batch)->GradInfoDict:
        grad_info = self._compute_grad_and_apply(train_batch)
        
        
        if hasattr(self, "stats_fn"):
            # for TorchPolicyV2
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

        return fetches


class TorchPolicyMod2:
    # fix grad_info missing issue
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
            batch_fetches[f"tower_{i}"].update(
                {
                    LEARNER_STATS_KEY: dict(**tower_outputs[i][1], **self.extra_grad_info(batch)),
                    "model": model.metrics(),
                }
            )

        batch_fetches.update(self.extra_compute_grad_fetches())

        return batch_fetches

    def _multi_gpu_parallel_grad_calc(
        self, sample_batches: List[SampleBatch]
    ) -> List[Tuple[List[TensorType], GradInfoDict]]:
        """Performs a parallelized loss and gradient calculation over the batch.

        Splits up the given train batch into n shards (n=number of this
        Policy's devices) and passes each data shard (in parallel) through
        the loss function using the individual devices' models
        (self.model_gpu_towers). Then returns each tower's outputs.

        Args:
            sample_batches: A list of SampleBatch shards to
                calculate loss and gradients for.

        Returns:
            A list (one item per device) of 2-tuples, each with 1) gradient
            list and 2) grad info dict.
        """
        assert len(self.model_gpu_towers) == len(sample_batches)
        lock = threading.Lock()
        results = {}
        grad_enabled = torch.is_grad_enabled()

        def _worker(shard_idx, model, sample_batch, device):
            torch.set_grad_enabled(grad_enabled)
            try:
                with NullContextManager() if device.type == "cpu" else torch.cuda.device(  # noqa: E501
                    device
                ):
                    loss_out = force_list(
                        self._loss(self, model, self.dist_class, sample_batch)
                    )

                    # Call Model's custom-loss with Policy loss outputs and
                    # train_batch.
                    loss_out = model.custom_loss(loss_out, sample_batch)

                    assert len(loss_out) == len(self._optimizers)

                    # Loop through all optimizers.
                    grad_info = {"allreduce_latency": 0.0}

                    parameters = list(model.parameters())
                    all_grads = [None for _ in range(len(parameters))]
                    for opt_idx, opt in enumerate(self._optimizers):
                        # Erase gradients in all vars of the tower that this
                        # optimizer would affect.
                        param_indices = self.multi_gpu_param_groups[opt_idx]
                        # for param_idx, param in enumerate(parameters):
                        #     if param_idx in param_indices and param.grad is not None:
                        #         param.grad.zero_()
                        opt.zero_grad()
                        # Recompute gradients of loss over all variables.
                        loss_out[opt_idx].backward()
                        grad_info.update(
                            self.extra_grad_process(opt, loss_out[opt_idx])
                        )

                        grads = []
                        # Note that return values are just references;
                        # Calling zero_grad would modify the values.
                        for param_idx, param in enumerate(parameters):
                            if param_idx in param_indices:
                                if param.grad is not None:
                                    grads.append(param.grad)
                                all_grads[param_idx] = param.grad

                        if self.distributed_world_size:
                            start = time.time()
                            if torch.cuda.is_available():
                                # Sadly, allreduce_coalesced does not work with
                                # CUDA yet.
                                for g in grads:
                                    torch.distributed.all_reduce(
                                        g, op=torch.distributed.ReduceOp.SUM
                                    )
                            else:
                                torch.distributed.all_reduce_coalesced(
                                    grads, op=torch.distributed.ReduceOp.SUM
                                )

                            for param_group in opt.param_groups:
                                for p in param_group["params"]:
                                    if p.grad is not None:
                                        p.grad /= self.distributed_world_size

                            grad_info["allreduce_latency"] += time.time() - \
                                start

                with lock:
                    results[shard_idx] = (all_grads, grad_info)
            except Exception as e:
                import traceback

                with lock:
                    results[shard_idx] = (
                        ValueError(
                            e.args[0]
                            + "\n traceback"
                            + traceback.format_exc()
                            + "\n"
                            + "In tower {} on device {}".format(shard_idx, device)
                        ),
                        e,
                    )

        # Single device (GPU) or fake-GPU case (serialize for better
        # debugging).
        if len(self.devices) == 1 or self.config["_fake_gpus"]:
            for shard_idx, (model, sample_batch, device) in enumerate(
                zip(self.model_gpu_towers, sample_batches, self.devices)
            ):
                _worker(shard_idx, model, sample_batch, device)
                # Raise errors right away for better debugging.
                last_result = results[len(results) - 1]
                if isinstance(last_result[0], ValueError):
                    raise last_result[0] from last_result[1]
        # Multi device (GPU) case: Parallelize via threads.
        else:
            threads = [
                threading.Thread(
                    target=_worker, args=(
                        shard_idx, model, sample_batch, device)
                )
                for shard_idx, (model, sample_batch, device) in enumerate(
                    zip(self.model_gpu_towers, sample_batches, self.devices)
                )
            ]

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        # Gather all threads' outputs and return.
        outputs = []
        for shard_idx in range(len(sample_batches)):
            output = results[shard_idx]
            if isinstance(output[0], Exception):
                raise output[0] from output[1]
            outputs.append(results[shard_idx])
        return outputs

