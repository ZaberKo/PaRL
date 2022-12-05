
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


from typing import List
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms import Algorithm
from ray.rllib.evaluation import SampleBatch
from ray.rllib.utils.typing import TensorType, TensorStructType
from ray.rllib.policy import Policy
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
    prev_states = [p.requires_grad for p in params]
    try:
        for param in params:
            param.requires_grad = False
        yield
    finally:
        for param, flag in zip(params, prev_states):
            param.requires_grad = flag


@contextlib.contextmanager
def print_time():
    start_time = time.time()
    try:
        yield
    finally:
        print(f"elapse time: {time.time()-start_time}")


def sample_from_sample_batch(sample_batch, size, keys=None):
    if sample_batch.get(SampleBatch.SEQ_LENS) is not None:
            raise ValueError(
                "SampleBatch.shuffle not possible when your data has "
                "`seq_lens` defined!"
            )

    if keys is None:
        keys = sample_batch.keys()


    idx = np.random.choice(len(sample_batch), size=size)
    print(idx)


    if keys is None:
        self_as_dict = {k: v for k, v in sample_batch.items()}
    else:
        self_as_dict = {k: v for k, v in sample_batch.items() if k in keys}
    # Note: adv index will create deep copy of the array
    sampled_dict = tree.map_structure(lambda v: v[idx], self_as_dict)

    new_sample_batch = SampleBatch(sampled_dict)

    return new_sample_batch


class CPUInitCallback(DefaultCallbacks):
    def on_algorithm_init(self, *, algorithm: Algorithm, **kwargs) -> None:
        num_cpus_for_local_worker = algorithm.config["num_cpus_for_driver"]
        num_cpus_for_rollout_worker = algorithm.config["num_cpus_per_worker"]
        # ============ driver worker multi-thread ==========
        # os.environ["OMP_NUM_THREADS"]=str(num_cpus_for_local_worker)
        # os.environ["OPENBLAS_NUM_THREADS"] = str(num_cpus_for_local_worker)
        # os.environ["MKL_NUM_THREADS"] = str(num_cpus_for_local_worker)
        # os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_cpus_for_local_worker)
        # os.environ["NUMEXPR_NUM_THREADS"] = str(num_cpus_for_local_worker)

        torch.set_num_threads(num_cpus_for_local_worker)

        # ============ rollout worker multi-thread ==========
        def set_rollout_num_threads(worker):
            torch.set_num_threads(num_cpus_for_rollout_worker)

        pendings = [w.apply.remote(set_rollout_num_threads)
                    for w in algorithm.workers.remote_workers()]
        ray.wait(pendings, num_returns=len(pendings))
