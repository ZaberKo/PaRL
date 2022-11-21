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


# class SharedNoiseTable:
#     def __init__(self, noise):
#         self.noise = noise
#         assert self.noise.dtype == np.float32

#     def get(self, i, dim):
#         return self.noise[i: i + dim]

#     def sample_index(self, dim):
#         return np.random.randint(0, len(self.noise) - dim + 1)


class NES(NeuroEvolution):
    """
        OpenAI-ES w/o some tricks
    """

    def __init__(self, config, pop_workers: WorkerSet, target_worker: RolloutWorker):
        super().__init__(config, pop_workers, target_worker)

        self.noise_stdev = config.get("noise_stdev", 0.05)
        self.step_size = config.get("step_size", 0.01)
        self.target_step_size = config.get("target_step_size", self.step_size)
        self.tau = config.get("pop_tau", 0.1)
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

        self.grad_norm = 0
        self.direction_norm = 0
        self.use_target_flag = False

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
        # self._evolve_always_with_smoothed_target(fitnesses, target_fitness)
        self._evolve_with_target_noise(fitnesses, target_fitness)

        # generate new pop
        self.generate_pop()
        self.sync_pop_weights()

    def _evolve_pop_only(self, fitnesses):
        fitnesses = np.asarray(fitnesses)
        ws = centered_ranks(fitnesses)

        grad = np.dot(ws, self.noise) / (self.noise_stdev * self.pop_size)
        self.grad_norm = np.linalg.norm(grad)

        self.mean, self.update_ratio = self.optimizer.update(-grad)

    def _evolve_with_target_noise(self, fitnesses, target_fitness):
        fitnesses.append(target_fitness)
        fitnesses = np.asarray(fitnesses)
        ws = centered_ranks(fitnesses)

        target_noise = (self.target_weights_flat-self.mean) / self.noise_stdev
        noise = np.concatenate([
            self.noise,
            np.expand_dims(target_noise, axis=0)
        ])

        grad = np.dot(ws, noise) / (self.noise_stdev * self.pop_size)
        self.grad_norm = np.linalg.norm(grad)

        self.mean, self.update_ratio = self.optimizer.update(-grad)

        self.use_target_flag = True

    def _evolve_with_target_dir(self, fitnesses, target_fitness):
        """
            Note: empirically not recommanded for failing tracking the target_weights
        """
        self._evolve_pop_only(fitnesses)

        self.use_target_flag = False

        if target_fitness > max(fitnesses):
            direction = self.target_weights_flat - self.mean
            self.direction_norm = np.linalg.norm(direction)
            # self.mean += self.target_step_size * \
            #     direction/np.linalg.norm(direction)
            self.mean += self.target_step_size * direction

            self.use_target_flag = True

    def _evolve_with_smoothed_target(self, fitnesses, target_fitness):
        """
            Note: empirically not recommanded for failing tracking the target_weights
        """
        self._evolve_pop_only(fitnesses)

        self.use_target_flag = False

        if target_fitness > max(fitnesses):
            self.mean = self.mean*(1-self.tau) + \
                self.target_weights_flat*self.tau
            self.use_target_flag = True

    def _evolve_always_with_smoothed_target(self, fitnesses, target_fitness):
        """
            Note: empirically not recommanded for failing tracking the target_weights
        """
        self._evolve_pop_only(fitnesses)

        self.mean = self.mean*(1-self.tau) + \
                self.target_weights_flat*self.tau
        self.use_target_flag = True

    @override(NeuroEvolution)
    def get_iteration_results(self):
        data = super().get_iteration_results()

        data.update({
            "target_pop_l2_distance": np.linalg.norm(self.target_weights_flat-self.mean, ord=2),
            "update_ratio": self.update_ratio,
            "grad_norm": self.grad_norm,
            # "dir_norm": self.direction_norm,
            "update_target_flag": self.use_target_flag
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
        state=super().save()

        state["mean"] = self.mean

        return state

    def restore(self, state):
        super().restore(state)

        for i in range(self.pop_size):
            self.pop_flat[i]=self.flatten_weights(self.pop[i])

        self.mean = state["mean"]


class NESPure(NES):
    @override(NES)
    def _evolve(self, fitnesses, target_fitness):
        self._evolve_pop_only(fitnesses)

        # generate new pop
        self.generate_pop()
        self.sync_pop_weights()

    @override(NES)
    def get_iteration_results(self):
        data = super(NES, self).get_iteration_results()

        return data