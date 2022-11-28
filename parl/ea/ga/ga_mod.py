import numpy as np
import ray

from parl.utils import ray_wait, clone_numpy_weights
from parl.ea.neuroevolution import NeuroEvolution
from .mutation import gaussian_mutate_inplace
from .crossover import crossover_inplace
from .selection import selection_tournament
from ray.rllib.utils.annotations import override


from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils.typing import ModelWeights
from typing import List, Dict

from .ga import GASelectionState


class GAMod(NeuroEvolution):
    def __init__(self, config, pop_workers: WorkerSet, target_worker: RolloutWorker):
        super().__init__(config, pop_workers, target_worker)

        self.migration_freq = config["migration_freq"]
        self.migration_start = config["migration_start"]

        self.elite_fraction = config["elite_fraction"]
        self.mutation_prob = config["mutation_prob"]
        self.mutation_std = config["mutation_std"]

        self.num_elitists = max(
            int(self.elite_fraction * self.pop_size), 1)

        pop = ray.get([
            worker.apply.remote(self.get_evolution_weights)
            for worker in self.pop_workers.remote_workers()
        ])
        # deep copy weights to make them writable
        self.pop = [clone_numpy_weights(i) for i in pop]

        self.target_selection_stat = GASelectionState()
        # last replaced id for target(worst id).
        self.target_id = None
        self.target_fitness = None

        self.best_id = None
        self.worst_id = None

    @override(NeuroEvolution)
    def _evolve(self, fitnesses: List[Dict], target_fitness):
        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
        # index from largest fitness to smallest
        index_rank = np.argsort(fitnesses)[::-1]
        elitists = index_rank[:self.num_elitists]  # Elitist indexes
        selects = selection_tournament(
            index_rank,
            num_offsprings=self.pop_size-self.num_elitists,
            tournament_size=3
        )

        self.best_id = index_rank[0]
        self.worst_id = index_rank[-1]

        # Figure out unselected candidates
        unselects = []
        for i in range(self.pop_size):
            if i in selects or i in elitists:
                continue
            else:
                unselects.append(i)
        np.random.shuffle(unselects)

        # if Lamarckian Transfer happens in prev iteration
        if self.target_id is not None:
            self.target_selection_stat.record(
                self.target_id, elitists, selects, unselects)
            self.target_fitness = fitnesses[self.target_id]

        # Elitism step, replace unselects by elites
        new_elitists = []
        for i in elitists:
            if len(unselects):
                replacee = unselects.pop(0)
            else:
                replacee = selects.pop(0)
            new_elitists.append(replacee)
            self.clone_pop_weights(src=i, dst=replacee)

        # Mutate all genes in the population except the new elitists
        for i in range(self.pop_size):
            if i not in new_elitists:  # Spare the new elitists
                if np.random.rand() < self.mutation_prob:
                    gaussian_mutate_inplace(self.pop[i], self.mutation_std)

        # delay sync_pop_weights() after RL

    def after_RL_training(self):
        # Lamarckian Transfer
        # Note: The execution order is little different from the original ERL
        if self.generation >= self.migration_start and self.generation % self.migration_freq == 0:
            target_weights = self.get_target_weights()
            target_weights = clone_numpy_weights(target_weights)
            self.target_id = self.worst_id
            self.pop[self.target_id] = target_weights
        else:
            self.target_id = None

        self.sync_pop_weights()

    def clone_pop_weights(self, src, dst):
        '''
            copy pop[src] to pop[dst] locally
        '''
        inv_src = self.pop[src]
        inv_dst = self.pop[dst]

        for name in inv_src.keys():
            np.copyto(inv_dst[name], inv_src[name])

    def get_iteration_results(self):
        data = super().get_iteration_results()

        data.update({
            "target_selection_rate": self.target_selection_stat.stats(),
            "best_id": self.best_id,
            "worst_id": self.worst_id,
            "target_fitness": self.target_fitness
        })

        return data
