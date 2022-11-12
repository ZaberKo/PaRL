import numpy as np
import torch
import torch.nn as nn


from ray.rllib.policy.torch_policy import TorchPolicy, _directStepOptimizerSingleton
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.evaluation import SampleBatch
from ray.rllib.utils.typing import (
    GradInfoDict,
    ModelWeights,
    TensorType,
)
from typing import List, Union, Dict, Tuple

TRUNCATED = "truncated"  # flag for sample_batch


def concat_multi_gpu_td_errors(
    policy: Union[TorchPolicy, TorchPolicyV2]
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
        "max_td_error": torch.max(td_error),
        "min_td_error": torch.min(td_error),
        "mean_td_error": torch.mean(td_error),
    }


def clip_and_record_grad_norm(optimizer, clip_value=None):
    grad_gnorm = 0

    if clip_value is None:
        clip_value = np.inf

    for param_group in optimizer.param_groups:
        grad_gnorm += nn.utils.clip_grad_norm_(
            param_group["params"], clip_value)

    return grad_gnorm.item()


# def get_trancated_info(
#     policy: Union[TorchPolicy, TorchPolicyV2], input_dict, state_batches, model, action_dist
# ) -> Dict[str, TensorType]:
#     infos = input_dict[SampleBatch.INFOS]
#     batch_size = infos.shape[0]

#     truncated = [False]*batch_size

#     # necessary for _initialize_loss_from_dummy_batch()
#     for i in range(batch_size):
#         if isinstance(infos[i], dict):
#             truncated = infos[i].get("TimeLimit.truncated", False)

#     return {
#         TRUNCATED: truncated
#     }


def postprocess_trancated_info(sample_batch: SampleBatch) -> SampleBatch:
    infos = sample_batch[SampleBatch.INFOS]

    truncateds = np.zeros_like(sample_batch[SampleBatch.DONES])

    for i, info in enumerate(infos):
        if isinstance(info, dict):
            if info.get("TimeLimit.truncated", False):
                truncateds[i] = True

    sample_batch[TRUNCATED] = truncateds

    return sample_batch
