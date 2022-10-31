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
from parl.utils import clone_numpy_weights

class GA(NeuroEvolution):
    def __init__(self, config, pop_workers: WorkerSet, target_worker: RolloutWorker):
        super().__init__(config, pop_workers, target_worker)
        self.gen = 0

        self.weight_magnitude = config["weight_magnitude_limit"]
        self.migration_freq = config["migration_freq"]
        self.migration_start = config["migration_start"]

        self.elite_fraction = config["elite_fraction"]
        self.crossover_prob = config["crossover_prob"]
        self.mutation_prob = config["mutation_prob"]

        self.num_elitists = max(
            int(self.elite_fraction * self.population_size), 1)

        self.pop = ray.get([
            worker.apply.remote(self.get_evolution_weights)
            for worker in self.pop_workers.remote_workers()
        ])

        # used to reduce communication cost
        self.update_flag = [False]*self.pop_size

    def _evolve(self, fitnesses: List[Dict]):
        self.gen += 1

        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
        # index from largest fitness to smallest
        index_rank = np.argsort(fitnesses)[::-1]
        elitists = index_rank[:self.num_elitists]  # Elitist indexes
        offsprings = selection_tournament(
            index_rank,
            num_offsprings=self.pop_size-self.num_elitists,
            tournament_size=3
        )

        # Figure out unselected candidates
        unselects = []
        for i in range(self.population_size):
            if i in offsprings or i in elitists:
                continue
            else:
                unselects.append(i)
        np.random.shuffle(unselects)

        # Elitism step, keep elites by copy them to unselects
        new_elitists = []
        for i in elitists:
            if len(unselects):
                replacee = unselects.pop(0)
            else:
                replacee = offsprings.pop(0)
            new_elitists.append(replacee)
            self.clone_pop_weights(src=i, dst=replacee)
            self.update_flag[replacee] = True

        # Crossover for unselected genes with 100 percent probability
        if len(unselects) % 2 != 0:
            # Number of unselects left should be even
            unselects.append(np.random.choice(unselects))
        for i, j in zip(unselects[0::2], unselects[1::2]):
            off_i = np.random.choice(new_elitists)
            off_j = np.random.choice(offsprings)

            self.clone_pop_weights(src=off_i, dst=i)  # copy off_i to i
            self.clone_pop_weights(src=off_j, dst=j)

            crossover_inplace(self.pop[i], self.pop[j])
            self.update_flag[i] = True
            self.update_flag[j] = True

        # Crossover for selected offsprings
        for i, j in zip(offsprings[0::2], offsprings[1::2]):
            if np.random.rand() < self.crossover_prob:
                crossover_inplace(self.pop[i], self.pop[j])
                self.update_flag[i] = True
                self.update_flag[j] = True

        # Mutate all genes in the population except the new elitists
        for i in range(self.population_size):
            if i not in new_elitists:  # Spare the new elitists
                if np.random.rand() < self.args.mutation_prob:
                    mutate_inplace(self.pop[i])
                    self.update_flag[i] = True

        if self.gen >= self.migration_start and self.gen % self.migration_freq == 0:
            with self.sync_pop_weights_timer:
                # TODO: check numpy share memory issue
                target_weights = self.get_evolution_weights(self.target_worker)
                target_weights = clone_numpy_weights(target_weights)
            replace_i = index_rank[-1]
            self.pop[replace_i] = target_weights
            self.update_flag[replace_i] = True

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

    def clone_pop_weights(self, src, dst):
        '''
            copy pop[src] to pop[dst] locally
        '''
        inv_src = self.pop[src]
        inv_dst = self.pop[dst]

        for name in inv_src.keys():
            np.copyto(inv_dst[name], inv_src[name])
