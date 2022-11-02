import numpy as np

from parl.ea.neuroevolution import NeuroEvolution

from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils.typing import ModelWeights

from ray.rllib.utils.annotations import override


class CEM(NeuroEvolution):
    def __init__(self, config, pop_workers: WorkerSet, target_worker: RolloutWorker):
        super().__init__(config, pop_workers, target_worker)

        self.num_elites = round(config["elite_fraction"]*self.pop_size)
        self.noise_decay_coeff = config["noise_decay_coeff"]
        self.noise_init = config["noise_init"]
        self.noise_end = config["noise_end"]

        # initialize pop
        target_weights = self.get_target_weights()
        self.params_shape = {}
        self.params_size = {}

        for name, param in target_weights.items():
            self.params_shape[name] = param.shape
            self.params_size[name] = param.size
        self.num_params = sum(self.params_size.values())

        self.mean = self.flatten_weights(target_weights)
        self.variance = np.full(self.num_params, self.noise_init)
        self.noise = self.noise_init

        self.pop_flat = np.zeros((self.pop_size, self.num_params))

        # static weights
        num_elites = self.num_elites+1  # include target
        _ws = np.log(1+num_elites)/np.arange(1, num_elites+1)
        self.ws = _ws/_ws.sum()

        self.generate_pop()
        self.sync_pop_weights()

        self.target_weights_flat = None

    def generate_pop(self):
        for i in range(self.pop_size):
            self.pop_flat[i] = np.random.normal(
                self.mean, np.sqrt(self.variance))

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
        self._evolve_target_compete_pop(fitnesses, target_fitness)
        # self._evolve_target_always_first(fitnesses, target_fitness)
        # self._evolve_pop_only(fitnesses)

    def _evolve_target_compete_pop(self, fitnesses, target_fitness):
        fitnesses.append(target_fitness)
        new_fitnesses = np.asarray(fitnesses)
        orders = new_fitnesses.argsort()[::-1]
        elite_ids = orders[:(self.num_elites+1)]

        target_weights = self.get_target_weights()
        # record target_weights_flat to calc the distance between pop mean
        self.target_weights_flat = self.flatten_weights(target_weights)

        pop_flat = np.concatenate([
            self.pop_flat,
            np.expand_dims(self.target_weights_flat, axis=0)
        ])

        elites = pop_flat[elite_ids]

        # update mean
        mean = np.dot(self.ws, elites)

        # update variance
        variance = np.full(self.num_params, self.noise) + np.dot(
            self.ws, np.power(elites-self.mean, 2))

        self.mean = mean
        self.variance = variance
        self.noise = self.noise_decay_coeff*self.noise + \
            (1-self.noise_decay_coeff)*self.noise_end

        # generate new pop
        self.generate_pop()
        self.sync_pop_weights()

    def _evolve_pop_only(self, fitnesses):
        fitnesses = np.asarray(fitnesses)
        orders = fitnesses.argsort()[::-1]
        elite_ids = orders[:(self.num_elites+1)]

        target_weights = self.get_target_weights()
        # record target_weights_flat to calc the distance between pop mean
        self.target_weights_flat = self.flatten_weights(target_weights)

        pop_flat = self.pop_flat

        elites = pop_flat[elite_ids]

        # update mean
        mean = np.dot(self.ws, elites)

        # update variance
        variance = np.full(self.num_params, self.noise) + np.dot(
            self.ws, np.power(elites-self.mean, 2))

        self.mean = mean
        self.variance = variance
        self.noise = self.noise_decay_coeff*self.noise + \
            (1-self.noise_decay_coeff)*self.noise_end

        # generate new pop
        self.generate_pop()
        self.sync_pop_weights()

    def _evolve_target_always_first(self, fitnesses):
        fitnesses = np.asarray(fitnesses)
        orders = fitnesses.argsort()
        elite_ids = orders[:self.num_elites]

        target_weights = self.get_target_weights()
        # record target_weights_flat to calc the distance between pop mean
        self.target_weights_flat = self.flatten_weights(target_weights)

        # update mean
        elites = np.concatenate([
            np.expand_dims(self.target_weights_flat, axis=0),
            self.pop_flat[elite_ids],
        ])
        mean = np.dot(self.ws, elites)

        # update variance
        variance = np.full(self.num_params, self.noise) + np.dot(
            self.ws, np.power(elites-self.mean, 2))

        self.mean = mean
        self.variance = variance
        self.noise = self.noise_decay_coeff*self.noise + \
            (1-self.noise_decay_coeff)*self.noise_end

        # generate new pop
        self.generate_pop()
        self.sync_pop_weights()

    @override(NeuroEvolution)
    def get_iteration_results(self):
        data = super().get_iteration_results()

        data.update({
            "var_noise": self.noise,
            "pop_var_mean": np.mean(self.variance),
            "pop_var_max": np.max(self.variance),
            "pop_var_min": np.min(self.variance),
            "target_pop_l2_distance": np.linalg.norm(self.target_weights_flat-self.mean, ord=2)
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

class CEMPure(CEM):
    @override(NeuroEvolution)
    def _evolve(self, fitnesses, target_fitness):
        # self._evolve_target_compete_pop(fitnesses, target_fitness)
        # self._evolve_target_always_first(fitnesses, target_fitness)
        self._evolve_pop_only(fitnesses)
