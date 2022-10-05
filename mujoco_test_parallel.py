# %%
import numpy as np
import gym
import ray
from ray.rllib.algorithms.sac import SAC, SACConfig

from ray.tune import Tuner, TuneConfig
from ray.air import RunConfig, CheckpointConfig

from policy import SACPolicy,SACPolicy_FixedAlpha
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
)
from ray.rllib.utils.metrics import SYNCH_WORKER_WEIGHTS_TIMER
from ray.rllib.execution.common import (
    LAST_TARGET_UPDATE_TS,
    NUM_TARGET_UPDATES,
)
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.replay_buffers.utils import sample_min_n_steps_from_buffer



num_test=1

num_rollout_workers=8
num_envs_per_worker=5

rollout_vs_train=4

config = SACConfig().framework('torch') \
    .rollouts(
        # rollout_fragment_length=1, # already set in SAC
        num_rollout_workers=num_rollout_workers,
        num_envs_per_worker=num_envs_per_worker,
        no_done_at_end=True,
        horizon=1000,
        soft_horizon=False)\
    .training(
        initial_alpha=1,
        train_batch_size=256,
        training_intensity=256/rollout_vs_train,
        replay_buffer_config={
        "_enable_replay_buffer_api": True,
        "type": "MultiAgentReplayBuffer",
        "capacity": int(1e6),
        # How many steps of the model to sample before learning starts.
        "learning_starts": 10000,
    })\
    .resources(num_gpus=0.1)\
    .evaluation(
        evaluation_interval=10, 
        evaluation_num_workers=16, 
        evaluation_duration=16,
        evaluation_config={
            "no_done_at_end":False,
            "horizon":None
    })\
    .reporting(
        min_time_s_per_iteration=None,
        min_sample_timesteps_per_iteration=1000, # 1000 updates per iteration
        metrics_num_episodes_for_smoothing=5
        ) \
    .environment(env="HalfCheetah-v3")\
    .to_dict()


def calculate_rr_weights(config: AlgorithmConfigDict):
    """Calculate the round robin weights for the rollout and train steps"""
    if not config["training_intensity"]:
        return [1, 1]

    # Calculate the "native ratio" as:
    # [train-batch-size] / [size of env-rolled-out sampled data]
    # This is to set freshly rollout-collected data in relation to
    # the data we pull from the replay buffer (which also contains old
    # samples).
    native_ratio = config["train_batch_size"] / (
        config["rollout_fragment_length"]
        * config["num_envs_per_worker"]
        # Add one to workers because the local
        # worker usually collects experiences as well, and we avoid division by zero.
        * max(config["num_workers"], 1)
    )

    # Training intensity is specified in terms of
    # (steps_replayed / steps_sampled), so adjust for the native ratio.
    sample_and_train_weight = config["training_intensity"] / native_ratio
    if sample_and_train_weight < 1:
        return [int(np.round(1 / sample_and_train_weight)), 1]
    else:
        return [1, int(np.round(sample_and_train_weight))]


class SAC_FixAlpha_Parallel(SAC):
    def get_default_policy_class(
        self, config):
        return SACPolicy_FixedAlpha
    
    def training_step(self) -> ResultDict:
        """DQN training iteration function.

        Each training iteration, we:
        - Sample (MultiAgentBatch) from workers.
        - Store new samples in replay buffer.
        - Sample training batch (MultiAgentBatch) from replay buffer.
        - Learn on training batch.
        - Update remote workers' new policy weights.
        - Update target network every `target_network_update_freq` sample steps.
        - Return all collected metrics for the iteration.

        Returns:
            The results dict from executing the training iteration.
        """
        train_results = {}

        # We alternate between storing new samples and sampling and training
        store_weight, sample_and_train_weight = calculate_rr_weights(self.config)

        for _ in range(store_weight):
            # Sample (MultiAgentBatch) from workers.
            new_sample_batch = synchronous_parallel_sample(
                worker_set=self.workers, concat=True
            )

            # Update counters
            self._counters[NUM_AGENT_STEPS_SAMPLED] += new_sample_batch.agent_steps()
            self._counters[NUM_ENV_STEPS_SAMPLED] += new_sample_batch.env_steps()

            # Store new samples in replay buffer.
            self.local_replay_buffer.add_batch(new_sample_batch)

        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }

        for _ in range(sample_and_train_weight):
            # Sample training batch (MultiAgentBatch) from replay buffer.
            train_batch = sample_min_n_steps_from_buffer(
                self.local_replay_buffer,
                self.config["train_batch_size"],
                count_by_agent_steps=self._by_agent_steps,
            )

            # Old-style replay buffers return None if learning has not started
            if train_batch is None or len(train_batch) == 0:
                self.workers.local_worker().set_global_vars(global_vars)
                break

            # Postprocess batch before we learn on it
            post_fn = self.config.get("before_learn_on_batch") or (lambda b, *a: b)
            train_batch = post_fn(train_batch, self.workers, self.config)

            # for policy_id, sample_batch in train_batch.policy_batches.items():
            #     print(len(sample_batch["obs"]))
            #     print(sample_batch.count)

            # Learn on training batch.
            # Use simple optimizer (only for multi-agent or tf-eager; all other
            # cases should use the multi-GPU optimizer, even if only using 1 GPU)
            if self.config.get("simple_optimizer") is True:
                train_results = train_one_step(self, train_batch)
            else:
                train_results = multi_gpu_train_one_step(self, train_batch)

            # Update replay buffer priorities.
            update_priorities_in_replay_buffer(
                self.local_replay_buffer,
                self.config,
                train_batch,
                train_results,
            )

            # Update target network every `target_network_update_freq` sample steps.
            cur_ts = self._counters[
                NUM_AGENT_STEPS_SAMPLED
                if self._by_agent_steps
                else NUM_ENV_STEPS_SAMPLED
            ]
            last_update = self._counters[LAST_TARGET_UPDATE_TS]
            if cur_ts - last_update >= self.config["target_network_update_freq"]:
                to_update = self.workers.local_worker().get_policies_to_train()
                self.workers.local_worker().foreach_policy_to_train(
                    lambda p, pid: pid in to_update and p.update_target()
                )
                self._counters[NUM_TARGET_UPDATES] += 1
                self._counters[LAST_TARGET_UPDATE_TS] = cur_ts

            # Update weights and global_vars - after learning on the local worker -
            # on all remote workers.
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                self.workers.sync_weights(global_vars=global_vars)

        # Return all collected metrics for the iteration.
        return train_results

#%%
calculate_rr_weights(config)
#%%

ray.init(num_cpus=(8+1+16)*num_test, num_gpus=1, local_mode=False, include_dashboard=True)
result_grid = Tuner(
    SAC_FixAlpha_Parallel,
    param_space=config,
    tune_config=TuneConfig(
        num_samples=num_test
    ),
    run_config=RunConfig(
        stop={"training_iteration": 1000}, # this will results in 1e6 updates
        checkpoint_config=CheckpointConfig(
            num_to_keep=None,  # save all checkpoints
            checkpoint_frequency=100
        )
    )
).fit()

import time
time.sleep(10)

# %%



# %%
