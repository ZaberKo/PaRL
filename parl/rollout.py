from itertools import chain

import ray
from ray.rllib.policy.sample_batch import concat_samples
from ray.rllib.evaluation.rollout_worker import RolloutWorker

from typing import List, Optional, Union
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils.typing import PolicyID, SampleBatchType, ModelGradients


def synchronous_parallel_sample(
    *,
    target_worker_set: WorkerSet,
    pop_worker_set: WorkerSet,
    episodes_per_worker: int = 1
) -> Union[List[SampleBatchType], SampleBatchType]:
    """Runs parallel and synchronous rollouts on all remote workers.

    Waits for all workers to return from the remote calls.

    If no remote workers exist (num_workers == 0), use the local worker
    for sampling.
    """

    num_target_workers = len(target_worker_set.remote_workers())
    num_pop_workers = len(pop_worker_set.remote_workers())

    def multiple_rollout(worker: RolloutWorker):
        return [worker.sample() for _ in range(episodes_per_worker)]

    # sample_batches axis: (worker, episode)
    if num_target_workers == 0:
        target_sample_batches = [
            target_worker_set.local_worker().apply(multiple_rollout)]
            
        pop_sample_batches = ray.get([
            worker.apply.remote(multiple_rollout)
            for worker in pop_worker_set.remote_workers()
        ])
    else:
        sample_batches = ray.get([
            worker.apply.remote(multiple_rollout)
            for worker in
            target_worker_set.remote_workers() +
            pop_worker_set.remote_workers()
        ])

        target_sample_batches = sample_batches[:num_target_workers]
        pop_sample_batches = sample_batches[num_target_workers:]

    return target_sample_batches, pop_sample_batches


def flatten_batches(sample_batches: List[List[SampleBatchType]]):
    return [batch for batches in sample_batches for batch in batches]
