import ray
import numpy as np
import torch
from parl.ea.neuroevolution import NeuroEvolution
from .utils import centered_ranks
from .optimizers import SGD, Adam
from parl.utils import ray_wait

from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils.typing import ModelWeights

from ray.rllib.utils.annotations import override
# from ray.rllib.utils.metrics.window_stat

# class SharedNoiseTable:
#     def __init__(self, noise):
#         self.noise = noise
#         assert self.noise.dtype == np.float32

#     def get(self, i, dim):
#         return self.noise[i: i + dim]

#     def sample_index(self, dim):
#         return np.random.randint(0, len(self.noise) - dim + 1)


class SafeES(NeuroEvolution):
    """
        Canonical ES (simplified CSA-ES) with safe muation
    """

    def __init__(self, config, pop_workers: WorkerSet, target_worker: RolloutWorker):
        super().__init__(config, pop_workers, target_worker)

        self.noise_stdev = config.get("noise_stdev", 0.05)
        self.parent_ratio = config.get("parent_ratio", 0.5)
        self.tau = config.get("pop_tau", 0.1)

        self.parent_size = round(self.pop_size*self.parent_ratio)

        # initialize pop
        target_weights = self.get_target_weights()
        self.params_shape = {}
        self.params_size = {}

        for name, param in target_weights.items():
            self.params_shape[name] = param.shape
            self.params_size[name] = param.size
        self.num_params = sum(self.params_size.values())

        # Note: the square of noise's l2 norm obey's chi-square dist
        # Therefore the expactation of noise l2 norm is
        self.noise_magnitude = np.sqrt(self.num_params)

        # no need to explicit deep copy
        self.mean = self.flatten_weights(target_weights)
        # record the last used noise
        self.noise = None

        self.pop_flat = np.zeros((self.pop_size, self.num_params))

        # static recombination weights. (lazy generate)
        self.ws = None

        self.generate_pop(init_pop=True)
        self.sync_pop_weights()

        self.target_weights_flat = None
        self.update_ratio = 1.0
        self.prev_target_weights_flat = self.mean.copy()

    def generate_pop(self, init_pop=False):
        noise = np.random.randn(self.pop_size, self.num_params)

        if init_pop:
            self.sensitivity = np.ones((self.pop_size, self.num_params))
        else:
            sensitivity = ray.get([
                worker.apply.remote(
                    self._get_mutation_sensitivity, mutation_batch_size=256)
                for worker in self.pop_workers.remote_workers()
            ])
            self.sensitivity = np.array(sensitivity)
            assert np.array_equal(self.sensitivity.shape, self.noise.shape)

        self.noise = self.noise_stdev * noise/self.sensitivity

        self.pop_flat = self.mean + self.noise

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
            weights[name] = weights_flat[pos:pos + size].reshape(
                self.params_shape[name])
            pos += size

        assert pos == self.num_params

        return weights

    @override(NeuroEvolution)
    def _evolve(self, fitnesses, target_fitness):
        # record target_weights_flat to calc the distance between pop mean
        self.target_weights_flat = self.flatten_weights(
            self.get_target_weights()
        )
        # self._evolve_pop_only(fitnesses)
        # self._evolve_with_target(fitnesses, target_fitness)
        # self._evolve_hybrid(fitnesses, target_fitness)
        # self._evolve_with_target_noise(fitnesses, target_fitness)
        # self._evolve_always_with_target_noise(fitnesses, target_fitness)
        # self._evolve_always_first_with_target_noise(fitnesses, target_fitness)
        self._evolve_always_first_with_target_noise(fitnesses, target_fitness)

        # generate new pop
        self.generate_pop()
        self.sync_pop_weights()

    def _evolve_with_target_noise(self, fitnesses, target_fitness):
        fitnesses.append(target_fitness)
        fitnesses = np.asarray(fitnesses)

        orders = fitnesses.argsort()[::-1]
        parent_ids = orders[:self.parent_size+1]

        target_noise = (self.target_weights_flat-self.mean) / self.noise_stdev
        noise = np.concatenate([
            self.noise,
            np.expand_dims(target_noise, axis=0)
        ])

        # use weighted recombination from CSA-ES
        if self.ws is None:
            parent_size = self.parent_size + 1
            _ws = np.log(parent_size+0.5)-np.log(np.arange(1, parent_size+1))
            self.ws = _ws/_ws.sum()

        self.mean += self.noise_stdev * \
            np.dot(self.ws, noise[parent_ids])


    def _evolve_always_with_target_noise(self, fitnesses, target_fitness):
        fitnesses = np.asarray(fitnesses)

        orders = fitnesses.argsort()[::-1]
        parent_ids = orders[:self.parent_size]

        # let target always be in the elite list:
        new_fitnesses = np.append(fitnesses[parent_ids], target_fitness)

        new_orders = [
            parent_ids[i] if i != len(new_fitnesses)-1 else len(fitnesses) for i in new_fitnesses.argsort()[::-1]
        ]

        target_noise = (self.target_weights_flat-self.mean)
        noise = np.concatenate([
            self.noise,
            np.expand_dims(target_noise, axis=0)
        ])

        # use weighted recombination from CSA-ES
        if self.ws is None:
            parent_size = self.parent_size + 1
            _ws = np.log(parent_size+0.5)-np.log(np.arange(1, parent_size+1))
            self.ws = _ws/_ws.sum()

        self.mean += np.dot(self.ws, noise[new_orders])


    def _evolve_always_first_with_target_noise(self, fitnesses, target_fitness):
        fitnesses = np.asarray(fitnesses)

        orders = fitnesses.argsort()[::-1]
        parent_ids = orders[:self.parent_size]

        target_noise = (self.target_weights_flat-self.mean) / self.noise_stdev
        noise = np.concatenate([
            np.expand_dims(target_noise, axis=0),
            self.noise[parent_ids],
        ])

        # use weighted recombination from CSA-ES
        if self.ws is None:
            parent_size = self.parent_size + 1
            _ws = np.log(parent_size+0.5)-np.log(np.arange(1, parent_size+1))
            self.ws = _ws/_ws.sum()

        self.mean += np.dot(self.ws, noise)



    @override(NeuroEvolution)
    def get_iteration_results(self):
        data = super().get_iteration_results()

        data.update({
            "target_pop_l2_distance": np.linalg.norm(self.target_weights_flat-self.mean, ord=2),
            "prev_target_pop_l2_distance": np.linalg.norm(self.prev_target_weights_flat-self.mean, ord=2),
            "sensitivity_min": np.min(self.sensitivity, axis=1),
            "sensitivity_max": np.max(self.sensitivity, axis=1)
        })

        return data

    def set_pop_weights(self, local_worker: RolloutWorker = None, remote_workers=None):
        weights = self.unflatten_weights(self.mean)

        if local_worker is not None:
            self.set_evolution_weights(
                worker=local_worker,
                weights=weights
            )

        if remote_workers is not None and len(remote_workers) > 0:
            weights_ref = ray.put(weights)
            ray_wait([
                worker.apply.remote(
                    self.set_evolution_weights, weights=weights_ref)
                for worker in remote_workers
            ])

    def save(self):
        state = super().save()

        state["mean"] = self.mean

        return state

    def restore(self, state):
        super().restore(state)

        for i in range(self.pop_size):
            self.pop_flat[i] = self.flatten_weights(self.pop[i])

        self.mean = state["mean"]

    @staticmethod
    def _get_mutation_sensitivity(worker: RolloutWorker, mutation_batch_size):
        return worker.get_policy().calc_sensitivity(mutation_batch_size)
