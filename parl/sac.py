import numpy as np
from ray.rllib.algorithms.sac import SAC, SACConfig

from parl.policy import SACPolicy

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
    TARGET_NET_UPDATE_TIMER
)
from ray.rllib.utils.metrics import SYNCH_WORKER_WEIGHTS_TIMER
from ray.rllib.execution.common import (
    LAST_TARGET_UPDATE_TS,
    NUM_TARGET_UPDATES
)


class SACConfigMod(SACConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or SAC_Parallel)
        self.tune_alpha = True
        self.policy_delay = 1

        # whether use huber_loss in critic loss
        self.use_huber = False
        self.huber_beta = 1.0

        self.optimization = {
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "entropy_learning_rate": 3e-4,
            "actor_grad_clip": None,
            "critic_grad_clip": None,
            "alpha_grad_clip": None,
            "critic_use_huber": False,
            "huber_beta": 1.0
        }

        self.q_model_config = {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
            "post_fcnet_hiddens": [],
            "post_fcnet_activation": None,
            "custom_model": None,  # Use this to define custom Q-model(s).
            "custom_model_config": {},
        }
        self.policy_model_config = {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
            "post_fcnet_hiddens": [],
            "post_fcnet_activation": None,
            "custom_model": None,  # Use this to define a custom policy model.
            "custom_model_config": {},
            "add_layer_norm": False
        }

    def training(
        self,
        *,
        add_actor_layer_norm: bool = None,
        tune_alpha: bool = None,
        policy_delay: int = None,
        use_huber: bool = None,
        huber_beta: float = None,
        optimization: dict = None,
        **kwargs
    ):
        super().training(**kwargs)
        if add_actor_layer_norm is not None:
            self.policy_model_config["add_layer_norm"] = add_actor_layer_norm

        if tune_alpha is not None:
            self.tune_alpha = tune_alpha
        if policy_delay is not None:
            self.policy_delay = policy_delay

        if use_huber is not None:
            self.use_huber = use_huber
        if huber_beta is not None:
            self.huber_beta = huber_beta

        if optimization is not None:
            self.optimization.update(optimization)

        return self


class SAC_Parallel(SAC):
    _allow_unknown_subkeys = SAC._allow_unknown_subkeys + \
        ["extra_python_environs_for_driver"]

    def get_default_policy_class(self, config):
        return SACPolicy

    @classmethod
    def get_default_config(cls) -> AlgorithmConfigDict:
        return SACConfigMod().to_dict()

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
        batch_size = self.config["train_batch_size"]
        local_worker = self.workers.local_worker()

        # Sample (MultiAgentBatch) from workers.
        new_sample_batches = synchronous_parallel_sample(
            worker_set=self.workers, concat=False
        )

        sampled_steps = 0
        for batch in new_sample_batches:
            # Update counters
            self._counters[NUM_AGENT_STEPS_SAMPLED] += batch.agent_steps()
            self._counters[NUM_ENV_STEPS_SAMPLED] += batch.env_steps()
            sampled_steps += batch.env_steps()
            # Store new samples in replay buffer.
            self.local_replay_buffer.add(batch)

        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }

        num_train_batches = sampled_steps
        for _ in range(num_train_batches):
            # Sample training batch (MultiAgentBatch) from replay buffer.
            train_batch = self.local_replay_buffer.sample(batch_size)

            # Old-style replay buffers return None if learning has not started
            if train_batch is None or len(train_batch) == 0:
                local_worker.set_global_vars(global_vars)
                break

            # Postprocess batch before we learn on it
            post_fn = self.config.get(
                "before_learn_on_batch") or (lambda b, *a: b)
            train_batch = post_fn(train_batch, self.workers, self.config)

            # for policy_id, sample_batch in train_batch.policy_batches.items():
            #     print(len(sample_batch["obs"]))
            #     print(sample_batch.count)

            # Learn on training batch.
            # Use simple optimizer (only for multi-agent or tf-eager; all other
            # cases should use the multi-GPU optimizer, even if only using 1 GPU)

            # train_results = multi_gpu_train_one_step(self, train_batch)
            train_results = train_one_step(self, train_batch)

            # Update replay buffer priorities.
            update_priorities_in_replay_buffer(
                self.local_replay_buffer,
                self.config,
                train_batch,
                train_results,
            )

            # Update target network every `target_network_update_freq` sample steps.
            cur_ts = self._counters[NUM_ENV_STEPS_SAMPLED]
            last_update = self._counters[LAST_TARGET_UPDATE_TS]
            if cur_ts - last_update >= self.config["target_network_update_freq"]:
                with self._timers[TARGET_NET_UPDATE_TIMER]:
                    to_update = local_worker.get_policies_to_train()
                    local_worker.foreach_policy_to_train(
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
