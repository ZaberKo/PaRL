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
        "max_td_error": torch.max(td_error),
        "min_td_error": torch.min(td_error),
        "mean_td_error": torch.mean(td_error),
    }

def clip_and_record_grad_norm(optimizer, clip_value=None):
    grad_gnorm = 0

    if clip_value is None:
        clip_value=np.inf

    for param_group in optimizer.param_groups:
        grad_gnorm+=nn.utils.clip_grad_norm_(param_group["params"], clip_value)

    return grad_gnorm.item()