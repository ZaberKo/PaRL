import numpy as np
import ray

from parl.utils import ray_wait, clone_numpy_weights
from parl.ea.neuroevolution import NeuroEvolution
from .mutation import mutate_inplace
from .crossover import crossover_inplace
from .selection import selection_tournament

from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils.typing import ModelWeights
from typing import List, Dict


class GA(NeuroEvolution):
    def __init__(self, config, pop_workers: WorkerSet, target_worker: RolloutWorker):
        super().__init__(config, pop_workers, target_worker)
        self.gen = 0

        self.weight_magnitude = config["weight_magnitude_limit"]
        self.migration_freq = config["migration_freq"]
        self.migration_start = config["migration_start"]

        self.elite_fraction = config["elite_fraction"]
        # self.crossover_prob=config["crossover_prob"]
        self.mutation_prob = config["mutation_prob"]

        self.num_elitists = max(
            int(self.elite_fraction * self.population_size), 2)

        self.pop = ray.get([
            worker.apply.remote(self.get_evolution_weights)
            for worker in self.pop_workers.remote_workers()
        ])

        # used to reduce communication cost
        self.update_flag = [False]*self.pop_size

    def evolve(self, fitnesses: List[Dict]):
        self.gen += 1

        
        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
        # index from largest fitness to smallest
        index_rank = np.argsort(fitnesses)[::-1]
        elitists = index_rank[:self.num_elitists]  # Elitist indexes safeguard

        with self.evolve_timer:
            if self.gen >= self.migration_start and self.gen % self.migration_freq == 0:
                with self.sync_pop_weights_timer:
                    # TODO: check numpy share memory issue
                    target_weights = self.get_evolution_weights(self.target_worker)
                replace_i = index_rank[-1]
                self.pop[replace_i] = target_weights
                self.update_flag[replace_i] = True

            # ===============================
            # Mutate all genes in the population except the elitists
            for net_i in index_rank[self.num_elitists:]:
                if np.random.rand() < self.mutation_prob:
                    mutate_inplace(self.pop[net_i], self.weight_magnitude)
                    self.update_flag[net_i] = True

        # send modified wieghts to remote pop_runners
        self.sync_pop_weights()

    def sync_pop_weights(self):
        with self.sync_pop_weights_timer:
            pendings = []

            for worker, weights, update_flag in zip(self.pop_workers.remote_workers(), self.pop, self.update_flag):
                if update_flag:
                    pendings.append(worker.apply.remote(
                        self.set_evolution_weights, weights=weights))
                    update_flag = False
            ray_wait(pendings)
