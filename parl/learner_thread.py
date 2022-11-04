from concurrent.futures import thread
import logging
import threading
import queue

from ray.util.timer import _Timer
from ray.rllib.evaluation import SampleBatch
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder, LEARNER_INFO
from ray.rllib.utils.metrics.window_stat import WindowStat
from ray.rllib.utils.framework import try_import_tf
from .utils import timer_to_ms

from ray.rllib.utils.metrics import (
    NUM_TARGET_UPDATES
)
from typing import Dict

tf1, tf, tfv = try_import_tf()

logger = logging.getLogger(__name__)


# Note: multiple _MultiGPULoaderThread are running in background.
class MultiGPULearnerThread(threading.Thread):
    def __init__(
            self,
            local_worker: RolloutWorker,
            target_network_update_freq = 1,
            num_multi_gpu_tower_stacks: int = 1,
            learner_queue_size: int = 16,
            num_data_load_threads: int = 16,
    ):
        threading.Thread.__init__(self)

        self.learner_queue_size = WindowStat("size", 50)
        self.local_worker = local_worker
        self.inqueue = queue.Queue(maxsize=learner_queue_size)
        self.outqueue = queue.Queue()

        self.queue_timer = _Timer()
        self.load_timer = _Timer()
        self.grad_timer = _Timer()
        self.load_wait_timer = _Timer()

        # setting daemon thread
        self.daemon = True

        self.weights_updated = False
        # record the last learner_info
        self.learner_info = {}
        self.stopped = False
        self.num_steps = 0

        self.policy_map = self.local_worker.policy_map
        self.devices = next(iter(self.policy_map.values())).devices
        logger.info("MultiGPULearnerThread devices {}".format(self.devices))
        self.tower_stack_indices = list(range(num_multi_gpu_tower_stacks))
        # Two queues for tower stacks:
        # a) Those that are loaded with data ("ready")
        # b) Those that are ready to be loaded with new data ("idle").
        self.idle_tower_stacks = queue.Queue()
        self.ready_tower_stacks = queue.Queue()
        # In the beginning, all stacks are idle (no loading has taken place
        # yet).
        for idx in self.tower_stack_indices:
            self.idle_tower_stacks.put(idx)
        # Start n threads that are responsible for loading data into the
        # different (idle) stacks.
        self.loader_threads = []
        for i in range(num_data_load_threads):
            loader_thread = _MultiGPULoaderThread(self, share_stats=(i == 0))
            loader_thread.start()
            self.loader_threads.append(loader_thread)


        # target model update metrics
        self.target_network_update_freq = target_network_update_freq
        self.num_updates_since_last_target_update = 0
        self.num_target_updates = 0
        self.target_net_update_timer= _Timer()

    def _update_target_networks(self):
        # Update target network every `target_network_update_freq` training update.
        self.num_updates_since_last_target_update += 1
        if self.num_updates_since_last_target_update > self.target_network_update_freq:
            with self.target_net_update_timer:
                to_update = self.local_worker.get_policies_to_train()
                self.local_worker.foreach_policy_to_train(
                    lambda p, pid: pid in to_update and p.update_target()
                )
            self.num_updates_since_last_target_update = 0
            self.num_target_updates += 1


    def run(self) -> None:
        # Switch on eager mode if configured.
        if self.local_worker.policy_config.get("framework") in ["tf2", "tfe"]:
            tf1.enable_eager_execution()
        while not self.stopped:
            self.step()

    def step(self) -> None:
        with self.load_wait_timer:
            while True:
                try:
                    buffer_idx = self.ready_tower_stacks.get(block=True)
                    break
                except queue.Empty:
                    logger.warn("Learner queue empty!")
                    # return _NextValueNotReady()

        num_steps_trained_this_iter = 0
        with self.grad_timer:
            # Use LearnerInfoBuilder as a unified way to build the final
            # results dict from `learn_on_loaded_batch` call(s).
            # This makes sure results dicts always have the same structure
            # no matter the setup (multi-GPU, multi-agent, minibatch SGD,
            # tf vs torch).
            learner_info_builder = LearnerInfoBuilder(
                num_devices=len(self.devices))

            for pid in self.policy_map.keys():
                # Not a policy-to-train.
                if not self.local_worker.is_policy_to_train(pid):
                    continue
                policy = self.policy_map[pid]

                logger.debug("== sgd update for {} ==".format(pid))

                # perform train_batch_size batch update
                default_policy_results = policy.learn_on_loaded_batch(
                    offset=0, buffer_index=buffer_idx
                )
                learner_info_builder.add_learn_on_batch_results(
                    default_policy_results)
                self.weights_updated = True
                num_steps_trained_this_iter += (
                    policy.get_num_samples_loaded_into_buffer(buffer_idx)
                )
            
            self.learner_info = learner_info_builder.finalize()

        self._update_target_networks()

        self.idle_tower_stacks.put(buffer_idx)

        self.outqueue.put((num_steps_trained_this_iter, self.learner_info))
        self.learner_queue_size.push(self.inqueue.qsize())

    def stats(self) -> Dict:
        """Add internal metrics to a result dict."""

        data = {
            "learner_queue": self.learner_queue_size.stats(),
            "learner_timer": {
                "learner_grad_time_ms": timer_to_ms(self.grad_timer),
                "learner_load_time_ms": timer_to_ms(self.load_timer),
                "learner_load_wait_time_ms": timer_to_ms(self.load_wait_timer),
                "learner_dequeue_time_ms": timer_to_ms(self.queue_timer),
                "target_net_update_time_ms": timer_to_ms(self.target_net_update_timer)
            },
            NUM_TARGET_UPDATES: self.num_target_updates,
        }

        return data


class _MultiGPULoaderThread(threading.Thread):
    def __init__(
        self, multi_gpu_learner_thread: MultiGPULearnerThread, share_stats: bool
    ):
        threading.Thread.__init__(self)
        self.multi_gpu_learner_thread = multi_gpu_learner_thread
        self.daemon = True
        if share_stats:
            self.queue_timer = multi_gpu_learner_thread.queue_timer
            self.load_timer = multi_gpu_learner_thread.load_timer
        else:
            self.queue_timer = _Timer()
            self.load_timer = _Timer()

    def run(self) -> None:
        while True:
            self.step()

    def step(self) -> None:
        s = self.multi_gpu_learner_thread
        policy_map = s.policy_map

        # Get a new batch from the data (inqueue).
        with self.queue_timer:
            batch = s.inqueue.get()

        # Get next idle stack for loading.
        buffer_idx = s.idle_tower_stacks.get()

        # Load the batch into the idle stack.
        with self.load_timer:
            for pid in policy_map.keys():
                if not s.local_worker.is_policy_to_train(pid, batch):
                    continue
                policy = policy_map[pid]
                if isinstance(batch, SampleBatch):
                    policy.load_batch_into_buffer(
                        batch=batch,
                        buffer_index=buffer_idx,
                    )
                elif pid in batch.policy_batches:
                    policy.load_batch_into_buffer(
                        batch=batch.policy_batches[pid],
                        buffer_index=buffer_idx,
                    )

        # Tag just-loaded stack as "ready".
        s.ready_tower_stacks.put(buffer_idx)
