from tqdm import trange
import logging
import copy
import math
import numpy as np


from ray.rllib.algorithms import Algorithm

from ray.rllib.evaluation import SampleBatch
from ray.rllib.evaluation.worker_set import WorkerSet

from ray.rllib.utils import merge_dicts
from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder
from ray.tune.execution.placement_groups import PlacementGroupFactory


from parl.rollout import synchronous_parallel_sample, flatten_batches
from parl.learner_thread import MultiGPULearnerThread
from parl.ea import NeuroEvolution, HybridES
from parl.parl_sac import PaRL_SAC
from parl.parl_td3 import PaRL_TD3
from parl.parl import (
    PaRL,
    make_learner_thread,
    evolver_algo,
    NUM_SAMPLES_ADDED_TO_QUEUE
)

from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_AGENT_STEPS_TRAINED,
    NUM_ENV_STEPS_TRAINED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.typing import (
    ResultDict,
    AlgorithmConfigDict,
    PartialAlgorithmConfigDict
)

logger = logging.getLogger(__name__)


class PaRL_HybridES:
    @override(Algorithm)
    def training_step(self) -> ResultDict:
        train_batch_size = self.config["train_batch_size"]
        local_worker = self.workers.local_worker()

        # Step 1: Sample episodes from workers.
        target_sample_batches, pop_sample_batches = synchronous_parallel_sample(
            target_worker_set=self.workers,
            pop_worker_set=self.pop_workers,
            episodes_per_worker=self.config["episodes_per_worker"]
        )

        # Step 2: Store samples into replay buffer
        sample_batches = flatten_batches(target_sample_batches) + \
            flatten_batches(pop_sample_batches)

        target_ts = sum([batch.env_steps() for batch in flatten_batches(target_sample_batches)])
        for batch in sample_batches:
            # Update sampling step counters.
            self._counters[NUM_ENV_STEPS_SAMPLED] += batch.env_steps()
            self._counters[NUM_AGENT_STEPS_SAMPLED] += batch.agent_steps()
            # Store new samples in the replay buffer
            # Use deprecated add_batch() to support old replay buffers for now
            self.local_replay_buffer.add(batch)

        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }


        # step 3: apply NE
        # Note: put EA ahead for correct target_weights
        if self.pop_size > 0:
            fitnesses = self._calc_fitness(pop_sample_batches)
            target_fitness = np.mean([episode[SampleBatch.REWARDS].sum(
            ) for episode in flatten_batches(target_sample_batches)])
            self.evolver.evolve(fitnesses, target_fitness=target_fitness)
            self.evolver.set_pop_weights(local_worker)


        # step 4: sample batches from replay buffer and place them on learner queue
        # num_train_batches = round(ts/train_batch_size*5)
        # num_train_batches = 1000
        num_train_batches = target_ts
        batch_bulk = self.config["batch_bulk"]

        real_num_train_batches = math.ceil(num_train_batches/batch_bulk)*batch_bulk

        # print("load train batch phase")
        for _ in range(math.ceil(num_train_batches/batch_bulk)):
            logger.info(f"add {num_train_batches} batches to learner thread")
            train_batch = self.local_replay_buffer.sample(
                train_batch_size*batch_bulk)

            # replay buffer learning start size not meet
            if train_batch is None or len(train_batch) == 0:
                self.workers.local_worker().set_global_vars(global_vars)
                return {}

            # target agent is updated at background thread
            self._learner_thread.inqueue.put(train_batch, block=True)
            self._counters[NUM_SAMPLES_ADDED_TO_QUEUE] += (
                batch.agent_steps() if self._by_agent_steps else batch.count
            )
        

        # Update replay buffer priorities.
        # update_priorities_in_replay_buffer(
        #     self.local_replay_buffer,
        #     self.config,
        #     train_batch,
        #     train_results,
        # )

        # step 5: retrieve train_results from learner thread and update target network
        train_results = self._retrieve_trained_results(real_num_train_batches)
        if self.pop_size > 0:
            train_results.update({
                "ea_results": self.evolver.get_iteration_results()
            })



        # step 6: generate offsprings
        self.evolver.resync_target_weights()
        self.evolver.generate_pop()
        self.evolver.sync_pop_weights()

        # step 7: sync target agent weights to rollout workers
        # Update weights and global_vars - after learning on the local worker - on all
        # remote workers.
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            self.workers.sync_weights(global_vars=global_vars)

        # Return all collected metrics for the iteration.
        return train_results


class PaRL_SAC_HybridES(PaRL_HybridES, PaRL_SAC):
    pass

class PaRL_TD3_HybridES(PaRL_HybridES, PaRL_TD3):
    pass