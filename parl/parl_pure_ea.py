from tqdm import trange
import logging
import copy
import platform
import math
import numpy as np

import ray

from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.sac import SAC
from parl.sac import SACConfigMod

from ray.rllib.evaluation import SampleBatch
from ray.rllib.evaluation.worker_set import WorkerSet
# from ray.rllib.execution.rollout_ops import synchronous_parallel_sample

from ray.rllib.utils.actors import create_colocated_actors
from ray.rllib.utils.replay_buffers.utils import update_priorities_in_replay_buffer
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder
from ray.tune.execution.placement_groups import PlacementGroupFactory

from ray.rllib.utils.metrics import (
    LAST_TARGET_UPDATE_TS,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_AGENT_STEPS_TRAINED,
    NUM_ENV_STEPS_TRAINED,
    NUM_TARGET_UPDATES,
    SYNCH_WORKER_WEIGHTS_TIMER,
    TARGET_NET_UPDATE_TIMER,
)
from parl.rollout import synchronous_parallel_sample, flatten_batches
from parl.learner_thread import MultiGPULearnerThread
from parl.ea import NeuroEvolution, CEM, ES, GA
from parl.policy import SACPolicy
from parl.parl import PaRLConfig

from ray.rllib.policy import Policy
from ray.rllib.utils.annotations import override
from ray.exceptions import GetTimeoutError, RayActorError, RayError
from ray.rllib.utils.typing import (
    ResultDict,
    AlgorithmConfigDict,
    PartialAlgorithmConfigDict,
    EnvType
)
from ray.tune.logger import Logger
from typing import (
    Callable,
    Container,
    DefaultDict,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

logger = logging.getLogger(__name__)

NUM_SAMPLES_ADDED_TO_QUEUE = "num_samples_added_to_queue"
SYNCH_POP_WORKER_WEIGHTS_TIMER = "synch_pop"

FITNESS = "fitness"

evolver_algo = {
    "es": ES,
    "ga": GA,
    "cem": CEM
}


class PaRL_PureEA(SAC):
    _allow_unknown_subkeys = SAC._allow_unknown_subkeys + \
        ["pop_config", "ea_config", "extra_python_environs_for_driver"]
    _override_all_subkeys_if_type_changes = SAC._override_all_subkeys_if_type_changes + \
        ["pop_config", "ea_config"]

    @override(Algorithm)
    def setup(self, config: PartialAlgorithmConfigDict):
        super().setup(config)

        self.pop_size = self.config["pop_size"]
        self.pop_config = merge_dicts(self.config, config["pop_config"])
        self.pop_workers = WorkerSet(
            env_creator=self.env_creator,
            validate_env=self.validate_env,
            policy_class=self.get_default_policy_class(self.pop_config),
            trainer_config=self.pop_config,
            num_workers=self.pop_size,
            local_worker=False,
            logdir=self.logdir,
        )
        if self.pop_size > 0:
            self.ea_config = self.config["ea_config"]
            evolver_cls = evolver_algo[self.config.get("evolver_algo", "cem")]
            self.evolver: NeuroEvolution = evolver_cls(
                self.ea_config, self.pop_workers, self.workers.local_worker())

        self.num_updates_since_last_target_update = 0

    @override(SAC)
    def training_step(self) -> ResultDict:
        train_batch_size = self.config["train_batch_size"]
        local_worker = self.workers.local_worker()

        # Step 1: Sample episodes from workers.
        target_sample_batches, pop_sample_batches = synchronous_parallel_sample(
            target_worker_set=None,
            pop_worker_set=self.pop_workers,
            episodes_per_worker=self.config["episodes_per_worker"]
        )

        assert len(target_sample_batches) == 0

        # Step 2: Store samples into replay buffer
        sample_batches = flatten_batches(target_sample_batches) + \
            flatten_batches(pop_sample_batches)

        # ts = 0  # total sample steps in the iteration
        for batch in sample_batches:
            # Update sampling step counters.
            self._counters[NUM_ENV_STEPS_SAMPLED] += batch.env_steps()
            self._counters[NUM_AGENT_STEPS_SAMPLED] += batch.agent_steps()
            # ts += batch.env_steps()
            # Store new samples in the replay buffer
            # Use deprecated add_batch() to support old replay buffers for now
            # self.local_replay_buffer.add(batch)

        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }

        # step 3: sample batches from replay buffer and place them on learner queue
        # num_train_batches = round(ts/train_batch_size*5)
        # num_train_batches = 1000
        # num_train_batches = round(ts/10)
        # for _ in range(num_train_batches):
        #     logger.info(f"add {num_train_batches} batches to learner thread")
        #     train_batch = self.local_replay_buffer.sample(train_batch_size)

        #     # replay buffer learning start size not meet
        #     if train_batch is None or len(train_batch) == 0:
        #         self.workers.local_worker().set_global_vars(global_vars)
        #         return {}

        #     # target agent is updated at background thread
        #     self._learner_thread.inqueue.put(train_batch, block=True)
        #     self._counters[NUM_SAMPLES_ADDED_TO_QUEUE] += (
        #         batch.agent_steps() if self._by_agent_steps else batch.count
        #     )

        # step 4: apply NE
        if self.pop_size > 0:
            fitnesses = self._calc_fitness(pop_sample_batches)
            target_fitness = np.mean([episode[SampleBatch.REWARDS].sum(
            ) for episode in flatten_batches(target_sample_batches)])
            self.evolver.evolve(fitnesses, target_fitness=target_fitness)
            # with self._timers[SYNCH_POP_WORKER_WEIGHTS_TIMER]:
            #     # set pop workers with new generated indv weights
            #     self.evolver.sync_pop_weights()

            # NEW:
            self.evolver.set_pop_weights(
                self.workers.local_worker()
            )

        # Update replay buffer priorities.
        # update_priorities_in_replay_buffer(
        #     self.local_replay_buffer,
        #     self.config,
        #     train_batch,
        #     train_results,
        # )

        # step 5: retrieve train_results from learner thread and update target network
        # train_results = self._process_trained_results()
        train_results = {}
        if self.pop_size > 0:
            train_results.update({
                "ea_results": self.evolver.get_iteration_results()
            })

        # step 6: sync target agent weights to rollout workers
        # Update weights and global_vars - after learning on the local worker - on all
        # remote workers.
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            self.workers.sync_weights(global_vars=global_vars)

        # Return all collected metrics for the iteration.
        return train_results



    def _calc_fitness(self, sample_batches):
        if self.pop_size != len(sample_batches):
            raise ValueError(
                "sample_batches must contains `pop_size` workers' batches")

        fitnesses = []
        for episodes in sample_batches:
            for episode in episodes:
                total_rewards = episode[SampleBatch.REWARDS].sum()
                fitnesses.append(total_rewards)
        return fitnesses



    @override(SAC)
    def validate_config(self, config: AlgorithmConfigDict) -> None:
        super().validate_config(config)

        if config["framework"] != "torch":
            raise ValueError("Current only support PyTorch!")

        # if config["num_workers"] <= 0:
        #     raise ValueError("`num_workers` for PaRL must be >= 1!")

        # if config["pop_size"] <= 0:
        #     raise ValueError("`pop_size` must be >=1")
        # elif round(config["pop_size"]*config["ea_config"]["elite_fraction"]) <= 0:
        #     raise ValueError(
        #         f'elite_fraction={config["elite_fraction"]} is too small with current pop_size={config["pop_size"]}.')

        if config["evaluation_interval"] <= 0:
            raise ValueError("evaluation_interval must >=1")

    @classmethod
    @override(SAC)
    def get_default_config(cls) -> AlgorithmConfigDict:
        return PaRLConfig().to_dict()

    @override(SAC)
    def get_default_policy_class(
        self, config: PartialAlgorithmConfigDict
    ) -> Optional[Type[Policy]]:
        return SACPolicy

    @classmethod
    @override(Algorithm)
    def default_resource_request(cls, config):
        config = dict(cls.get_default_config(), **config)
        pop_config = merge_dicts(config, config["pop_config"])
        eval_config = merge_dicts(config, config["evaluation_config"])

        bundles = []

        # driver worker
        bundles += [
            {
                "CPU": config["num_cpus_for_driver"],
                "GPU": 0 if config["_fake_gpus"] else config["num_gpus"]
            }
        ]

        # target_workers
        bundles += [
            {
                # RolloutWorkers.
                "CPU": config["num_cpus_per_worker"],
                "GPU": config["num_gpus_per_worker"],
                **config["custom_resources_per_worker"],
            }
            for _ in range(config["num_workers"])
        ]

        # pop_workers
        bundles += [
            {
                # RolloutWorkers.
                "CPU": pop_config["num_cpus_per_worker"],
                "GPU": pop_config["num_gpus_per_worker"],
                **pop_config["custom_resources_per_worker"],
            }
            for _ in range(config["pop_size"])
        ]

        # eval_workers
        bundles += [
            {
                # RolloutWorkers.
                "CPU": eval_config["num_cpus_per_worker"],
                "GPU": eval_config["num_gpus_per_worker"],
                **eval_config["custom_resources_per_worker"],
            }
            for _ in range(config["evaluation_num_workers"])
        ]

        return PlacementGroupFactory(bundles=bundles, strategy=config.get("placement_strategy", "PACK"))
