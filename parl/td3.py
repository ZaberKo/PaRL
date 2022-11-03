from ray.rllib.algorithms.td3 import TD3, TD3Config
from parl.policy.td3_policy import TD3Policy

from ray.rllib.execution.rollout_ops import (
    synchronous_parallel_sample,
)
from ray.rllib.execution.train_ops import (
    train_one_step,
    multi_gpu_train_one_step,
)
from ray.rllib.utils.replay_buffers.utils import update_priorities_in_replay_buffer
from ray.rllib.utils.typing import (
    ResultDict,
    AlgorithmConfigDict,
)
from ray.rllib.utils.metrics import (
    NUM_ENV_STEPS_SAMPLED,
    NUM_AGENT_STEPS_SAMPLED,
    TARGET_NET_UPDATE_TIMER,
)
from ray.rllib.utils.metrics import SYNCH_WORKER_WEIGHTS_TIMER
from ray.rllib.execution.common import (
    LAST_TARGET_UPDATE_TS,
    NUM_TARGET_UPDATES,
)

class TD3ConfigMod(TD3Config):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or TD3Mod)

        self.add_actor_layer_norm = False

    def training(
        self,
        *,
        add_actor_layer_norm: bool = None,
        **kwargs
    ):
        super().training(**kwargs)
        if add_actor_layer_norm is not None:
            self.add_actor_layer_norm = add_actor_layer_norm

        return self

class TD3Mod(TD3):
    def get_default_policy_class(self, config):
        return TD3Policy

    def training_step(self) -> ResultDict:
        """Simple Q training iteration function.

        Simple Q consists of the following steps:
        - Sample n MultiAgentBatches from n workers synchronously.
        - Store new samples in the replay buffer.
        - Sample one training MultiAgentBatch from the replay buffer.
        - Learn on the training batch.
        - Update the target network every `target_network_update_freq` sample steps.
        - Return all collected training metrics for the iteration.

        Returns:
            The results dict from executing the training iteration.
        """
        batch_size = self.config["train_batch_size"]
        local_worker = self.workers.local_worker()

        # Sample n MultiAgentBatches from n workers.
        new_sample_batches = synchronous_parallel_sample(
            worker_set=self.workers, concat=False
        )

        sampled_steps=0
        for batch in new_sample_batches:
            # Update sampling step counters.
            self._counters[NUM_ENV_STEPS_SAMPLED] += batch.env_steps()
            self._counters[NUM_AGENT_STEPS_SAMPLED] += batch.agent_steps()
            # Store new samples in the replay buffer
            # Use deprecated add_batch() to support old replay buffers for now
            self.local_replay_buffer.add(batch)
            sampled_steps+=batch.env_steps()

        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }

        num_train_batches=sampled_steps
        for _ in range(num_train_batches):
            # Use deprecated replay() to support old replay buffers for now
            train_batch = self.local_replay_buffer.sample(batch_size)
            # If not yet learning, early-out here and do not perform learning, weight-
            # synching, or target net updating.
            if train_batch is None or len(train_batch) == 0:
                self.workers.local_worker().set_global_vars(global_vars)
                return {}

            # Learn on the training batch.
            # Use simple optimizer (only for multi-agent or tf-eager; all other
            # cases should use the multi-GPU optimizer, even if only using 1 GPU)
            train_results = train_one_step(self, train_batch)


            # Update replay buffer priorities.
            update_priorities_in_replay_buffer(
                self.local_replay_buffer,
                self.config,
                train_batch,
                train_results,
            )

            # TODO: Move training steps counter update outside of `train_one_step()` method.
            # # Update train step counters.
            # self._counters[NUM_ENV_STEPS_TRAINED] += train_batch.env_steps()
            # self._counters[NUM_AGENT_STEPS_TRAINED] += train_batch.agent_steps()

            # Update target network every `target_network_update_freq` sample steps.
            cur_ts = self._counters[
                NUM_AGENT_STEPS_SAMPLED if self._by_agent_steps else NUM_ENV_STEPS_SAMPLED
            ]
            last_update = self._counters[LAST_TARGET_UPDATE_TS]
            if cur_ts - last_update >= self.config["target_network_update_freq"]:
                with self._timers[TARGET_NET_UPDATE_TIMER]:
                    to_update = local_worker.get_policies_to_train()
                    local_worker.foreach_policy_to_train(
                        lambda p, pid: pid in to_update and p.update_target()
                    )
                self._counters[NUM_TARGET_UPDATES] += 1
                self._counters[LAST_TARGET_UPDATE_TS] = cur_ts

        # Update weights and global_vars - after learning on the local worker - on all
        # remote workers.
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            self.workers.sync_weights(global_vars=global_vars)

        # Return all collected metrics for the iteration.
        return train_results