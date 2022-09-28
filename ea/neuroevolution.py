import ray
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.evaluation import RolloutWorker


from utils import ray_wait, clone_numpy_weights


from ray.actor import ActorHandle
from ray.rllib.utils.typing import ModelWeights

from typing import List

from ray.util.timer import _Timer
from utils import timer_to_ms


class NeuroEvolution:
    def __init__(self, config, pop_workers: WorkerSet, target_workers: WorkerSet):
        self.config = config  # ea_config
        self.pop_workers = pop_workers
        self.target_workers = target_workers

        self.pop_size = len(pop_workers.remote_workers())
        self.pop = [None]*self.pop_size
        self.generation = 0

        self.evolve_timer = _Timer()
        self.load_target_weights_timer = _Timer()
        self.sync_pop_weights_timer = _Timer()

    def evolve(self, fitnesses):
        raise NotImplementedError

    def sync_pop_weights(self):
        with self.sync_pop_weights_timer:
            pendings = []

            for worker, weights in zip(self.pop_workers.remote_workers(), self.pop):
                pendings.append(worker.apply.remote(
                    self.set_evolution_weights, weights=weights))
            ray_wait(pendings)

    def stats(self):
        data = {
            "generation": self.generation,
            "evolution_timer": {
                "evolve_time_ms": timer_to_ms(self.evolve_timer),
                "load_target_weights_time_ms": timer_to_ms(self.load_target_weights_timer),
                "sync_pop_weights_time_ms": timer_to_ms(self.sync_pop_weights_timer)
            }
        }

        return data

    @staticmethod
    def get_evolution_weights(worker: RolloutWorker) -> ModelWeights:
        policy = worker.get_policy()
        weights = policy.get_evolution_weights()
        return weights

    @staticmethod
    def set_evolution_weights(worker: RolloutWorker, weights: ModelWeights):
        policy = worker.get_policy()
        policy.set_evolution_weights(weights)
