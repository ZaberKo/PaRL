from typing import Iterable, List
import copy
import numpy as np
import ray
import tree
import time
import importlib
import contextlib

from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.framework import try_import_torch


from typing import Dict, Optional
from ray.rllib.utils.typing import TensorType, TensorStructType
from ray.rllib.policy import Policy
from ray.rllib.evaluation import SampleBatch
from ray.util.timer import _Timer
torch, _ = try_import_torch()


def ray_wait(pendings: List):
    '''
        wait all pendings without timeout
    '''
    return ray.wait(pendings, num_returns=len(pendings))[0]


def clone_numpy_weights(x: TensorStructType,):
    def mapping(item):
        if isinstance(item, np.ndarray):
            ret = item.copy()  # copy to avoid sharing (make it writeable)
        else:
            ret = item
        return ret

    return tree.map_structure(mapping, x)


def timer_to_ms(timer: _Timer):
    return round(1000 * timer.mean, 3)


def compute_ranks(x):
    """Returns ranks in [0, len(x))

    Note: This is different from scipy.stats.rankdata, which returns ranks in
    [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks

# Note: use from_config() instead


def import_policy_class(policy_name) -> Policy:
    # read policy_class from config
    tmp = policy_name.rsplit('.', 1)
    if len(tmp) == 2:
        module, name = tmp
        return getattr(importlib.import_module(module), name)
    else:
        raise ValueError('`policy_name` is incorrect')


def disable_grad(params: list[torch.Tensor]):
    for param in params:
        param.requires_grad = False


def enable_grad(params: list[torch.Tensor]):
    for param in params:
        param.requires_grad = False

@contextlib.contextmanager
def disable_grad_ctx(params: list[torch.Tensor]):
    prev_states=[p.requires_grad for p in params]
    try:
        for param in params:
            param.requires_grad = False
        yield
    finally:
        for param, flag in zip(params, prev_states):
            param.requires_grad = flag


@contextlib.contextmanager
def print_time():
    start_time=time.time()
    try:
        yield 
    finally:
        print(f"elapse time: {time.time()-start_time}")