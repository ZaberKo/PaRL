from tqdm import trange
import logging
import copy
import math
import numpy as np


from ray.rllib.algorithms import Algorithm

from ray.rllib.evaluation import SampleBatch
from ray.rllib.evaluation.worker_set import WorkerSet

from ray.rllib.utils import merge_dicts
from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder
from ray.tune.execution.placement_groups import PlacementGroupFactory


from parl.rollout import synchronous_parallel_sample, flatten_batches
from parl.learner_thread import MultiGPULearnerThread
from parl.ea import NeuroEvolution, CEM, ES, GA, CEMPure, HybridES

from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_AGENT_STEPS_TRAINED,
    NUM_ENV_STEPS_TRAINED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.typing import (
    ResultDict,
    AlgorithmConfigDict,
    PartialAlgorithmConfigDict
)

logger = logging.getLogger(__name__)

NUM_SAMPLES_ADDED_TO_QUEUE = "num_samples_added_to_queue"
# SYNCH_POP_WORKER_WEIGHTS_TIMER = "synch_pop"

FITNESS = "fitness"

evolver_algo = {
    "es": ES,
    "ga": GA,
    "cem": CEM,
    "cem-pure": CEMPure,
    "hybrid-es": HybridES
}


def make_learner_thread(local_worker, config):
    logger.info(
        "Enabling multi-GPU mode, {} GPUs, {} parallel tower-stacks".format(
            config["num_gpus"], config["num_multi_gpu_tower_stacks"]
        )
    )

    learner_thread = MultiGPULearnerThread(
        local_worker=local_worker,
        target_network_update_freq=config["target_network_update_freq"],
        batch_bulk=config["batch_bulk"],
        num_multi_gpu_tower_stacks=config["num_multi_gpu_tower_stacks"],
        learner_queue_size=config["learner_queue_size"],
        num_data_load_threads=config["num_data_load_threads"]
    )

    return learner_thread


class PaRLBaseConfig:
    def __init__(self) -> None:
        # flag for pop worker:
        self.is_pop_worker = False

        self.episodes_per_worker = 1
        # EA config
        self.pop_size = 10
        self.pop_config = {
            "explore": False,
            "batch_mode": "complete_episodes",
            "rollout_fragment_length": 1
        }
        self.evolver_algo = 'cem'
        self.ea_config = {
            "elite_fraction": 0.5,
            "noise_decay_coeff": 0.95,
            "noise_init": 1e-3,
            "noise_end": 1e-5
        }

        # learner thread config
        self.num_multi_gpu_tower_stacks = 8
        self.learner_queue_size = 16
        self.num_data_load_threads = 16
        self.batch_bulk = 1

        self.target_network_update_freq = 1  # unit: iteration

        # reporting
        self.metrics_episode_collection_timeout_s = 60.0
        self.metrics_num_episodes_for_smoothing = 5
        self.min_time_s_per_iteration = 0
        self.min_sample_timesteps_per_iteration = 0
        self.min_train_timesteps_per_iteration = 0

        # default_resources
        self.num_cpus_per_worker = 1
        self.num_envs_per_worker = 1
        self.num_cpus_for_local_worker = 1
        self.num_gpus_per_worker = 0

        self.framework("torch")


class PaRL:
    _allow_unknown_subkeys = Algorithm._allow_unknown_subkeys + \
        ["pop_config", "ea_config", "extra_python_environs_for_driver"]
    _override_all_subkeys_if_type_changes = Algorithm._override_all_subkeys_if_type_changes + \
        ["pop_config", "ea_config"]

    @override(Algorithm)
    def setup(self, config: PartialAlgorithmConfigDict):
        super(PaRL, self).setup(config)

        self.pop_size = self.config["pop_size"]
        self.pop_config = merge_dicts(self.config, config["pop_config"])
        self.pop_config["is_pop_worker"] = True
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

    @override(Algorithm)
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

        target_ts = sum([batch.env_steps() for batch in flatten_batches(target_sample_batches)])
        for batch in sample_batches:
            # Update sampling step counters.
            self._counters[NUM_ENV_STEPS_SAMPLED] += batch.env_steps()
            self._counters[NUM_AGENT_STEPS_SAMPLED] += batch.agent_steps()
            # Store new samples in the replay buffer
            # Use deprecated add_batch() to support old replay buffers for now
            self.local_replay_buffer.add(batch)

        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }

        # step 3: apply NE
        # Note: put EA ahead for correct target_weights
        if self.pop_size > 0:
            fitnesses = self._calc_fitness(pop_sample_batches)
            target_fitness = np.mean([episode[SampleBatch.REWARDS].sum(
            ) for episode in flatten_batches(target_sample_batches)])
            self.evolver.evolve(fitnesses, target_fitness=target_fitness)

        # step 4: sample batches from replay buffer and place them on learner queue
        # Note: training and target network update are happened in learner_thread

        # num_train_batches = round(ts/train_batch_size*5)
        # num_train_batches = 1000
        num_train_batches = target_ts
        batch_bulk = self.config["batch_bulk"]

        real_num_train_batches = math.ceil(num_train_batches/batch_bulk)*batch_bulk

        # print("load train batch phase")
        for _ in range(math.ceil(num_train_batches/batch_bulk)):
            logger.info(f"add {num_train_batches} batches to learner thread")
            train_batch = self.local_replay_buffer.sample(
                train_batch_size*batch_bulk)

            # replay buffer learning start size not meet
            if train_batch is None or len(train_batch) == 0:
                self.workers.local_worker().set_global_vars(global_vars)
                return {}

            # target agent is updated at background thread
            self._learner_thread.inqueue.put(train_batch, block=True)
            self._counters[NUM_SAMPLES_ADDED_TO_QUEUE] += (
                batch.agent_steps() if self._by_agent_steps else batch.count
            )

        # Update replay buffer priorities.
        # update_priorities_in_replay_buffer(
        #     self.local_replay_buffer,
        #     self.config,
        #     train_batch,
        #     train_results,
        # )

        # step 5: retrieve train_results from learner thread
        train_results = self._retrieve_trained_results(real_num_train_batches)
        if self.pop_size > 0:
            train_results.update({
                "ea_results": self.evolver.get_iteration_results()
            })

        # step 6: sync target agent weights to its rollout workers
        # Update weights and global_vars - after learning on the local worker - on all
        # remote workers.
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            self.workers.sync_weights(global_vars=global_vars)

        # Return all collected metrics for the iteration.
        return train_results

    def _retrieve_trained_results(self, num_train_batches) -> ResultDict:
        # Get learner outputs/stats from output queue.
        # and update the target network
        # print("backgroud training phase")
        learner_infos = []
        num_env_steps_trained = 0
        num_agent_steps_trained = 0
        # Loop through output queue and update our counts.
        # for _ in range(self._learner_thread.outqueue.qsize()):
        for _ in range(num_train_batches):
            if self._learner_thread.is_alive():
                (
                    env_steps,
                    learner_results,
                ) = self._learner_thread.outqueue.get()
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

    def _calc_fitness(self, sample_batches):
        if self.pop_size != len(sample_batches):
            raise ValueError(
                "sample_batches must contains `pop_size` workers' batches")

        fitnesses = []
        for episodes in sample_batches:
            total_rewards = 0
            for episode in episodes:
                total_rewards += episode[SampleBatch.REWARDS].sum()
            total_rewards /= len(episodes)
            fitnesses.append(total_rewards)
        return fitnesses

    @override(Algorithm)
    def _compile_iteration_results(self, *args, **kwargs):
        result = super(PaRL, self)._compile_iteration_results(*args, **kwargs)
        # add learner thread metrics
        result["info"].update(
            self._learner_thread.stats()
        )
        if self.pop_size > 0:
            result["info"].update(
                self.evolver.stats()
            )
        return result

    @override(Algorithm)
    def validate_config(self, config: AlgorithmConfigDict) -> None:
        super(PaRL, self).validate_config(config)

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
