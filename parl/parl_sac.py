from tqdm import trange
import logging
import copy
import numpy as np

import ray

from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.sac import SAC
from parl.sac import SACConfigMod
from parl.parl import PaRL

from ray.rllib.evaluation import SampleBatch
from ray.rllib.evaluation.worker_set import WorkerSet

from ray.rllib.utils import merge_dicts
from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder
from ray.tune.execution.placement_groups import PlacementGroupFactory


from parl.policy import SACPolicy

from ray.rllib.policy import Policy
from ray.rllib.utils.annotations import override

from ray.rllib.utils.typing import (
    ResultDict,
    AlgorithmConfigDict,
    PartialAlgorithmConfigDict
)
from typing import (
    Optional,
    Type
)

logger = logging.getLogger(__name__)



class PaRLSACConfig(SACConfigMod):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or PaRL)

        self.store_buffer_in_checkpoints = False
        self.replay_buffer_config = {
            "type": "MultiAgentReplayBuffer",
            "capacity": int(1e6),
            "learning_starts": 10000,
            # "no_local_replay_buffer": True,
            "replay_buffer_shards_colocated_with_driver": False,
            # "num_replay_buffer_shards": 1
        }

        self.optimization.update({
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "entropy_learning_rate": 3e-4,
        })

        self.policy_model_config.update({
            "add_layer_norm": True
        })

        self.episodes_per_worker = 1

        # EA config
        self.pop_size = 10
        self.pop_config = {
            # "explore": True,
            # "batch_mode": "complete_episodes",
            # "rollout_fragment_length": 1
        }

        self.evolver_algo = 'cem'
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
        self.tune_alpha = True

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
        self.min_train_timesteps_per_iteration = 0

        # default_resources
        self.num_cpus_per_worker = 1
        self.num_envs_per_worker = 1
        self.num_cpus_for_local_worker = 1
        self.num_gpus_per_worker = 0

        self.framework("torch")





class PaRL_SAC(PaRL, SAC):
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
        return PaRLSACConfig().to_dict()

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
