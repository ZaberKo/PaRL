import numpy as np

from ea.neuroevolution import NeuroEvolution
from utils import compute_ranks

from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils.typing import ModelWeights

from ray.rllib.utils.annotations import override


class CEM(NeuroEvolution):
    def __init__(self, config, pop_workers: WorkerSet, target_workers: WorkerSet):
        NeuroEvolution.__init__(self, config, pop_workers, target_workers)

        self.num_elites = round(config["elite_fraction"]*self.pop_size)
        self.noise_decay_coeff = config["noise_decay_coeff"]
        self.noise_init = config["noise_init"]
        self.noise_end = config["noise_end"]

        # initialize pop
        local_target_worker = self.target_workers.local_worker()
        target_weights = self.get_evolution_weights(local_target_worker)
        self.params_shape = {}
        self.params_size = {}

        for name, param in self.mean.items():
            self.params_shape[name] = param.shape
            self.params_size[name] = param.size
        self.num_params = sum(self.params_size.values())

        self.mean = self.flatten_weights(target_weights)
        self.variance = np.full(self.num_params, self.noise_init)
        self.noise = self.noise_init

        self.pop_flat = np.zeros((self.pop_size, self.num_params))

        # static weights
        _ws = np.log(2+self.num_elites)/np.arange(1, self.num_elites+2)
        self.ws = _ws/_ws.sum()

        self.generate_pop()
        self.sync_pop_weights()

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
        for param in weights.values:
            param_list.append(param.flatten())

        weights_flat = np.concatenate(param_list)

        assert len(weights_flat) == self.num_params

        return weights_flat

    def unflatten_weights(self, weights_flat: np.ndarray) -> ModelWeights:
        pos = 0
        weights = {}
        for name, size in self.params_size.items():
            weights[name] = weights_flat[pos:pos+size].reshape(self.params_shape[name])
            pos += size

        assert pos == self.num_params

        return weights

    def evolve(self, fitnesses):
        fitnesses = np.asarray(fitnesses)
        orders = fitnesses.argsort()
        elite_ids=orders[:self.num_elites]

        with self.load_target_weights_timer:
            target_weights = self.get_evolution_weights(
                self.target_workers.local_worker())
            target_weights_flat = self.flatten_weights(target_weights)

        with self.evolve_timer:
            # update mean
            mean = target_weights_flat * self.ws[0] + np.dot(self.pop_flat[elite_ids],self.ws[1:])

            # update variance
            variance=np.full(self.num_params,self.noise) + np.dot(
                np.power(self.pop_flat[elite_ids]-self.mean,2),
                self.ws[1:])
            
            self.mean=mean
            self.variance=variance
            self.noise=self.noise_decay_coeff*self.noise + (1-self.noise_decay_coeff)*self.noise_end

        # generate new pop
        self.generate_pop()
        self.sync_pop_weights()
            