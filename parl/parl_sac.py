from tqdm import trange
import logging
import copy
import numpy as np

import ray

from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.sac import SAC
from parl.sac import SACConfigMod
from parl.parl import PaRL, PaRLBaseConfig

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



class PaRLSACConfig(PaRLBaseConfig, SACConfigMod):
    def __init__(self, algo_class=None):
        SACConfigMod.__init__(self, algo_class=algo_class or PaRL)

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

        # training config
        self.n_step = 1
        self.initial_alpha = 1.0
        self.tau = 0.005
        self.normalize_actions = False
        self.policy_delay = 1
        self.tune_alpha = True

        PaRLBaseConfig.__init__(self)


class PaRL_SAC(PaRL, SAC):
    @classmethod
    @override(SAC)
    def get_default_config(cls) -> AlgorithmConfigDict:
        return PaRLSACConfig().to_dict()

    @override(SAC)
    def get_default_policy_class(
        self, config: PartialAlgorithmConfigDict
    ) -> Optional[Type[Policy]]:
        return SACPolicy
