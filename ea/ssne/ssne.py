import numpy as np
import ray
from ray.rllib.evaluation import metrics
from .mutation import mutate_inplace
from .crossover import crossover_inplace
from .selection import selection_tournament
from ray.rllib.utils.typing import ModelWeights

from utils import ray_wait, clone_numpy_weights
from typing import List, Dict


class EAStat:
    def __init__(self) -> None:
        self.elite = 0
        self.selected = 0
        self.discarded = 0
        self.n = 0
        self.num_migrations = 0

    def stats(self):
        return {
            "num_migrations": self.num_migrations,
            # "elite_rate": self.elite/self.n,
            # "selection_rate": self.selected/self.n,
            # "discard_rate": self.discarded/self.n
        }


class SSNE:
    def __init__(self, config, pop_runners, target_runner,shm_sync=True):
        self.gen = 0

        self.population_size = config["pop_size"]
        self.weight_magnitude = config["weight_magnitude_limit"]
        self.migration_freq = config["migration_freq"]
        self.migration_start = config["migration_start"]

        self.elite_fraction=config["elite_fraction"]
        # self.crossover_prob=config["crossover_prob"]
        self.mutation_prob=config["mutation_prob"]

        self.num_elitists = max(int(self.elite_fraction * self.population_size), 2)

        pop_refs = [
            worker.apply.remote(self.get_learning_weights)
            for worker in self.pop_workers.remote_workers()
        ]
        # break the share memory connect. make a local copy
        self.pop: List[ModelWeights]  = [clone_numpy_weights(w) for w in ray.get(pop_refs)]

        self.update_flag=[False]*len(self.pop)

        self.state=EAStat()

        #TODO: enable shm_sync function

    def evolve(self, fitnesses: List[Dict]):
        evolve_metrics = {}

        self.gen += 1

        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
        index_rank = np.argsort(fitnesses)[::-1] # index from largest fitness to smallest
        elitists = index_rank[:self.num_elitists]  # Elitist indexes safeguard

        if self.gen>=self.migration_start and self.gen % self.migration_freq == 0:
            target=get_model_weights(self.target_runner)
            replace_i = index_rank[-1]
            hard_update(dst=self.pop[replace_i], src=target)
            self.state.num_migrations += 1
            self.update_flag[replace_i]=True

        # ===============================
        # Mutate all genes in the population except the elitists
        for net_i in index_rank[self.num_elitists:]:
            if np.random.rand() < self.mutation_prob:
                mutate_inplace(self.pop[net_i], self.weight_magnitude)
                self.update_flag[net_i]=True

        # send modified wieghts to remote pop_runners
        self.sync_pop_weights()

        evolve_metrics["evolve_generation"] = self.gen
        evolve_metrics.update(self.state.stats())

        return evolve_metrics


    def sync_pop_weights(self):
        # TODO: !! currently no need for explicitly sync call on single machine since the weights are shared through shared memory.
        for i,(r,w) in enumerate(zip(self.pop_runners,self.pop)):
            if self.update_flag[i]:
                # Note: The weights_sync is non-blocking
                set_model_weights(r,w)
                self.update_flag[i]=False


def hard_update(dst: ModelWeights, src: ModelWeights):
    """Hard update (clone) from target network to source: weights and buffers
    """

    # note: the input type is OrderedDict
    # We assume the keys are same, and do not check for them
    for target_param, param in zip(dst.values(), src.values()):
        np.copyto(target_param,param)
        # target_param.copy_(param)
        