from parl.parl import PaRL

from parl.policy import TD3Policy
from parl.td3 import TD3ConfigMod
from ray.rllib.algorithms.td3 import TD3

from ray.rllib.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import (
    AlgorithmConfigDict,
    PartialAlgorithmConfigDict
)
from typing import Optional, Type

class PaRLTD3Config(TD3ConfigMod):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or PaRL_TD3)

        self.add_actor_layer_norm = True
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

        self.evolver_algo = 'cem'

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

class PaRL_TD3(PaRL, TD3):
    @classmethod
    @override(TD3)
    def get_default_config(cls) -> AlgorithmConfigDict:
        return PaRLTD3Config().to_dict()

    @override(TD3)
    def get_default_policy_class(
        self, config: PartialAlgorithmConfigDict
    ) -> Optional[Type[Policy]]:
        return TD3Policy