from tqdm import trange
import logging
import copy
import platform
import math
import numpy as np

import ray

from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.sac import SAC, SACConfig

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
from .rollout import synchronous_parallel_sample, flatten_batches
from .learner_thread import MultiGPULearnerThread
from .ea import NeuroEvolution, CEM
from .policy import SACPolicy

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


class PaRLConfig(SACConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or PaRL)

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
        }

        self.store_buffer_in_checkpoints = False
        self.replay_buffer_config = {
            "type": "MultiAgentReplayBuffer",
            "capacity": int(1e6),
            "learning_starts": 50000,
            # "no_local_replay_buffer": True,
            "replay_buffer_shards_colocated_with_driver": False,
            # "num_replay_buffer_shards": 1
        }

        self.optimization = {
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "entropy_learning_rate": 3e-4,
        }

        self.episodes_per_worker = 1

        # EA config
        self.pop_size = 10
        self.pop_config = {
            # "explore": True,
            # "batch_mode": "complete_episodes",
            # "rollout_fragment_length": 1
        }

        self.ea_config = {
            "elite_fraction": 0.5,
            "noise_decay_coeff": 0.95,
            "noise_init": 1e-3,
            "noise_end": 1e-5
        }

        # training config
        self.n_step = 1
        self.initial_alpha = 1.0
        self.tau = 0.005
        self.normalize_actions = True
        self.policy_delay = 1

        # learner thread config
        self.num_multi_gpu_tower_stacks = 8
        self.learner_queue_size = 16
        self.num_data_load_threads = 16

        self.target_network_update_freq = 1  # unit: iteration

        # reporting
        self.metrics_episode_collection_timeout_s = 60.0
        self.metrics_num_episodes_for_smoothing = 5
        self.min_time_s_per_iteration = 0
        self.min_sample_timesteps_per_iteration = 0
        self.min_train_timesteps_per_iteration = 256*1000

        # default_resources
        self.num_cpus_per_worker = 1
        self.num_envs_per_worker = 1
        self.num_cpus_for_local_worker = 1
        self.num_gpus_per_worker = 0

        self.framework("torch")


def make_learner_thread(local_worker, config):
    logger.info(
        "Enabling multi-GPU mode, {} GPUs, {} parallel tower-stacks".format(
            config["num_gpus"], config["num_multi_gpu_tower_stacks"]
        )
    )

    learner_thread = MultiGPULearnerThread(
        local_worker=local_worker,
        num_multi_gpu_tower_stacks=config["num_multi_gpu_tower_stacks"],
        learner_queue_size=config["learner_queue_size"],
        num_data_load_threads=config["num_data_load_threads"]
    )

    return learner_thread


class PaRL(SAC):
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

        self.ea_config = self.config["ea_config"]
        self.evolver: NeuroEvolution = CEM(
            self.ea_config, self.pop_workers, self.workers.local_worker())

        # ========== remote replay buffer ========

        # # Create copy here so that we can modify without breaking other logic
        # replay_actor_config = copy.deepcopy(self.config["replay_buffer_config"])
        # num_replay_buffer_shards = replay_actor_config.get("num_replay_buffer_shards",1)
        # replay_actor_config["capacity"] = (
        #     self.config["replay_buffer_config"]["capacity"] // num_replay_buffer_shards
        # )
        # ReplayActor = ray.remote(num_cpus=0)(replay_actor_config["type"])

        # if replay_actor_config["replay_buffer_shards_colocated_with_driver"]:
        #     self._replay_actors = create_colocated_actors(
        #         actor_specs=[  # (class, args, kwargs={}, count)
        #             (
        #                 ReplayActor,
        #                 None,
        #                 replay_actor_config,
        #                 num_replay_buffer_shards,
        #             )
        #         ],
        #         node="localhost",  # localhost
        #     )[0]  # [0]=only one item in `actor_specs`.
        # # Place replay buffer shards on any node(s).
        # else:
        #     self._replay_actors = [
        #         ReplayActor.remote(*replay_actor_config)
        #         for _ in range(num_replay_buffer_shards)
        #     ]

        # =========== learner thread ===========
        # do not sync pop_workers weights
        self._learner_thread = make_learner_thread(
            self.workers.local_worker(), self.config)
        self._learner_thread.start()

        self.num_updates_since_last_target_update = 0

    @override(SAC)
    def training_step(self) -> ResultDict:
        train_batch_size = self.config["train_batch_size"]
        local_worker = self.workers.local_worker()

        # Step 1: Sample episodes from workers.
        target_sample_batches, pop_sample_batches = synchronous_parallel_sample(
            target_worker_set=self.workers,
            pop_worker_set=self.pop_workers,
            episodes_per_worker=self.config["episodes_per_worker"]
        )

        # Step 2: Store samples into replay buffer
        sample_batches = flatten_batches(target_sample_batches) + \
            flatten_batches(pop_sample_batches)


        ts = 0 # total sample steps in the iteration
        for batch in sample_batches:
            # Update sampling step counters.
            self._counters[NUM_ENV_STEPS_SAMPLED] += batch.env_steps()
            self._counters[NUM_AGENT_STEPS_SAMPLED] += batch.agent_steps()
            ts += batch.env_steps()
            # Store new samples in the replay buffer
            # Use deprecated add_batch() to support old replay buffers for now
            self.local_replay_buffer.add(batch)

        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }

        # step 3: sample batches from replay buffer and place them on learner queue
        # num_train_batches = round(ts/train_batch_size)
        # num_train_batches = 1000
        num_train_batches = ts
        for _ in trange(num_train_batches):
            logger.info(f"add {num_train_batches} batches to learner thread")
            train_batch = self.local_replay_buffer.sample(train_batch_size)

            # replay buffer learning start size not meet
            if train_batch is None or len(train_batch) == 0:
                self.workers.local_worker().set_global_vars(global_vars)
                return {}

            # target agent is updated at background thread
            self._learner_thread.inqueue.put(train_batch, block=True)
            self._counters[NUM_SAMPLES_ADDED_TO_QUEUE] += (
                batch.agent_steps() if self._by_agent_steps else batch.count
            )

        # step 4: apply NE
        fitnesses = self._calc_fitness(pop_sample_batches)
        target_fitness = np.mean([episode[SampleBatch.REWARDS].sum() for episode in flatten_batches(target_sample_batches)])
        self.evolver.evolve(fitnesses, target_fitness=target_fitness)
        with self._timers[SYNCH_POP_WORKER_WEIGHTS_TIMER]:
            # set pop workers with new generated indv weights
            self.evolver.sync_pop_weights()

        # Update replay buffer priorities.
        # update_priorities_in_replay_buffer(
        #     self.local_replay_buffer,
        #     self.config,
        #     train_batch,
        #     train_results,
        # )

        # step 5: retrieve train_results from learner thread and update target network
        train_results = self._process_trained_results()
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

    def _process_trained_results(self) -> ResultDict:
        # Get learner outputs/stats from output queue.
        # and update the target network
        learner_infos = []
        num_env_steps_trained = 0
        num_agent_steps_trained = 0
        # Loop through output queue and update our counts.
        for _ in range(self._learner_thread.outqueue.qsize()):
            if self._learner_thread.is_alive():
                (
                    env_steps,
                    learner_results,
                ) = self._learner_thread.outqueue.get()
                self._update_target_networks()
                num_env_steps_trained += env_steps
                num_agent_steps_trained += env_steps
                if learner_results:
                    learner_infos.append(learner_results)
            else:
                raise RuntimeError("The learner thread died while training")
        # Nothing new happened since last time, use the same learner stats.
        if not learner_infos:
            final_learner_info = copy.deepcopy(
                self._learner_thread.learner_info)
        # Accumulate learner stats using the `LearnerInfoBuilder` utility.
        else:
            builder = LearnerInfoBuilder()
            for info in learner_infos:
                builder.add_learn_on_batch_results_multi_agent(info)
            final_learner_info = builder.finalize()
        # Update the steps trained counters.
        self._counters[NUM_ENV_STEPS_TRAINED] += num_env_steps_trained
        self._counters[NUM_AGENT_STEPS_TRAINED] += num_agent_steps_trained
        return final_learner_info

    def _update_target_networks(self):
        local_worker = self.workers.local_worker()
        # Update target network every `target_network_update_freq` training update.
        self.num_updates_since_last_target_update += 1
        if self.num_updates_since_last_target_update >= self.config["target_network_update_freq"]:
            with self._timers[TARGET_NET_UPDATE_TIMER]:
                to_update = local_worker.get_policies_to_train()
                local_worker.foreach_policy_to_train(
                    lambda p, pid: pid in to_update and p.update_target()
                )
            self.num_updates_since_last_target_update = 0
            self._counters[NUM_TARGET_UPDATES] += 1
            self._counters[LAST_TARGET_UPDATE_TS] = self._counters[
                NUM_AGENT_STEPS_TRAINED
                if self._by_agent_steps
                else NUM_ENV_STEPS_TRAINED
            ]

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

    @override(Algorithm)
    def _compile_iteration_results(self, *args, **kwargs):
        result = super()._compile_iteration_results(*args, **kwargs)
        # add learner thread metrics
        result["info"].update(
            self._learner_thread.stats()
        )
        result["info"].update(
            self.evolver.stats()
        )
        return result

    @override(SAC)
    def validate_config(self, config: AlgorithmConfigDict) -> None:
        super().validate_config(config)

        if config["framework"] != "torch":
            raise ValueError("Current only support PyTorch!")

        # if config["num_workers"] <= 0:
        #     raise ValueError("`num_workers` for PaRL must be >= 1!")

        if config["pop_size"] <= 0:
            raise ValueError("`pop_size` must be >=1")
        elif round(config["pop_size"]*config["ea_config"]["elite_fraction"]) <= 0:
            raise ValueError(
                f'elite_fraction={config["elite_fraction"]} is too small with current pop_size={config["pop_size"]}.')

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
