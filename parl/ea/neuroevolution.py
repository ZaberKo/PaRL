import ray
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.evaluation import RolloutWorker
from ray.util.timer import _Timer

from parl.utils import (
    ray_wait, 
    clone_numpy_weights, 
    timer_to_ms
)

from ray.actor import ActorHandle
from ray.rllib.utils.typing import ModelWeights


class NeuroEvolution:
    def __init__(self, config, pop_workers: WorkerSet, target_worker: RolloutWorker):
        self.config = config  # ea_config
        self.pop_workers = pop_workers
        self.target_worker = target_worker

        self.pop_size = len(pop_workers.remote_workers())
        self.pop = [None]*self.pop_size
        self.generation = 0

        self.evolve_timer = _Timer()
        self.load_target_weights_timer = _Timer()
        self.sync_pop_weights_timer = _Timer()

        # record last iteration fitnesses and target weights
        self.fitnesses = None
        self.target_weights = None

    def evolve(self, fitnesses, **kwargs):
        self.fitnesses = fitnesses
        self.generation += 1
        with self.evolve_timer:
            self._evolve(fitnesses.copy(), **kwargs)

    def _evolve(self, fitnesses, **kwargs):
        raise NotImplementedError

    def get_target_weights(self):
        with self.load_target_weights_timer:
            self.target_weights = self.get_evolution_weights(
                self.target_worker)

        return self.target_weights

    def sync_pop_weights(self):
        with self.sync_pop_weights_timer:
            pendings = []

            for worker, weights in zip(self.pop_workers.remote_workers(), self.pop):
                pendings.append(worker.apply.remote(
                    self.set_evolution_weights, weights=weights))
            ray_wait(pendings)

    def after_RL_training(self):
        """
            callback function: EA->RL->this
        """
        pass

    def stats(self):
        data = {
            "evolution_timer": {
                "evolve_time_ms": timer_to_ms(self.evolve_timer),
                "load_target_weights_time_ms": timer_to_ms(self.load_target_weights_timer),
                "sync_pop_weights_time_ms": timer_to_ms(self.sync_pop_weights_timer)
            }
        }

        return data

    def get_iteration_results(self):
        return {
            "generation": self.generation,
            "fitness": self.fitnesses
        }

    def save(self):
        return {
            "pop": self.pop
        }

    def restore(self, state):
        self.pop = state["pop"]

    @staticmethod
    def get_evolution_weights(worker: RolloutWorker) -> ModelWeights:
        policy = worker.get_policy()
        weights = policy.get_evolution_weights()
        return weights

    @staticmethod
    def set_evolution_weights(worker: RolloutWorker, weights: ModelWeights):
        policy = worker.get_policy()
        policy.set_evolution_weights(weights)
