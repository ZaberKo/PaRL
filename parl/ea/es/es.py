import ray
import numpy as np
import torch
from parl.ea.neuroevolution import NeuroEvolution
from .utils import centered_ranks
from .optimizers import SGD, Adam

from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils.typing import ModelWeights

from ray.rllib.utils.annotations import override


# class SharedNoiseTable:
#     def __init__(self, noise):
#         self.noise = noise
#         assert self.noise.dtype == np.float32

#     def get(self, i, dim):
#         return self.noise[i: i + dim]

#     def sample_index(self, dim):
#         return np.random.randint(0, len(self.noise) - dim + 1)


class ES(NeuroEvolution):
    """
        paper: Back to basics: Benchmarking canonical evolution strategies for playing atari
    """

    def __init__(self, config, pop_workers: WorkerSet, target_worker: RolloutWorker):
        super().__init__(config, pop_workers, target_worker)

        self.noise_stdev = config.get("noise_stdev", 0.05)
        self.step_size = config.get("step_size", 0.01)
        self.target_step_size = config.get("target_step_size", self.step_size)
        es_optim = config.get("es_optim", 'sgd')

        # initialize pop
        target_weights = self.get_target_weights()
        self.params_shape = {}
        self.params_size = {}

        for name, param in target_weights.items():
            self.params_shape[name] = param.shape
            self.params_size[name] = param.size
        self.num_params = sum(self.params_size.values())

        # no need to explicit deep copy
        self.mean = self.flatten_weights(target_weights)
        # record the last used noise
        self.noise = None

        self.pop_flat = np.zeros((self.pop_size, self.num_params))

        # _ws = np.log(self.pop_size+0.5)-np.log(np.arange(1, self.pop_size+1))
        # self.ws = _ws/_ws.sum()
        if es_optim == "adam":
            self.optimizer = Adam(theta=self.mean, stepsize=self.step_size)
        elif es_optim == 'sgd':
            self.optimizer = SGD(theta=self.mean, stepsize=self.step_size)
        else:
            raise ValueError('es_optim must be "sgd" or "adam"')

        self.generate_pop()
        self.sync_pop_weights()

        self.target_weights_flat = None
        self.update_ratio = 1.0

    def generate_pop(self):
        self.noise = np.random.randn(self.pop_size, self.num_params)

        self.pop_flat = self.mean + self.noise_stdev * self.noise

    @override(NeuroEvolution)
    def sync_pop_weights(self):
        for i in range(self.pop_size):
            self.pop[i] = self.unflatten_weights(self.pop_flat[i])

        super().sync_pop_weights()

    def flatten_weights(self, weights: ModelWeights) -> np.ndarray:
        # only need learnable weights in policy(actor)
        param_list = []
        for param in weights.values():
            param_list.append(param.flatten())

        weights_flat = np.concatenate(param_list)

        assert len(weights_flat) == self.num_params

        return weights_flat

    def unflatten_weights(self, weights_flat: np.ndarray) -> ModelWeights:
        pos = 0
        weights = {}
        for name, size in self.params_size.items():
            weights[name] = weights_flat[pos:pos +
                                         size].reshape(self.params_shape[name])
            pos += size

        assert pos == self.num_params

        return weights

    @override(NeuroEvolution)
    def _evolve(self, fitnesses, target_fitness):
        self._evolve_pop_only(fitnesses)
        # self._evolve_with_target(fitnesses, target_fitness)

    def _evolve_pop_only(self, fitnesses):
        fitnesses = np.asarray(fitnesses)

        ws = centered_ranks(fitnesses)

        grad = np.dot(ws, self.noise) / (self.noise_stdev * self.pop_size)

        self.mean, self.update_ratio = self.optimizer(-grad)

        # generate new pop
        self.generate_pop()
        self.sync_pop_weights()

    def _evolve_with_target(self, fitnesses, target_fitness):
        fitnesses = np.asarray(fitnesses)
        ws = centered_ranks(fitnesses)

        grad = np.dot(ws, self.noise) / (self.noise_stdev * self.pop_size)

        self.mean, self.update_ratio = self.optimizer(-grad)


        if target_fitness > max(fitnesses):
            target_weights = self.get_target_weights()
            self.target_weights_flat = self.flatten_weights(target_weights)

            direction = self.target_weights_flat - target_weights
            self.mean += self.target_step_size * \
                direction/self.np.linalg.norm(direction)

    @override(NeuroEvolution)
    def get_iteration_results(self):
        data = super().get_iteration_results()

       
        target_weights = self.get_target_weights()
        # record target_weights_flat to calc the distance between pop mean
        self.target_weights_flat = self.flatten_weights(target_weights)

        data.update({
            "target_pop_l2_distance": np.linalg.norm(self.target_weights_flat-self.mean, ord=2),
            "update_ratio": self.update_ratio
        })

        return data

    def set_pop_weights(self, worker: RolloutWorker = None):
        # Note: use for local worker
        if worker is None:
            worker = self.target_worker.local_worker()

        self.set_evolution_weights(
            worker=worker,
            weights=self.unflatten_weights(self.mean)
        )
