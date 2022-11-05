from tqdm import trange
import logging

from ray.rllib.evaluation.worker_set import WorkerSet
# from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.tune.execution.placement_groups import PlacementGroupFactory

from parl.ea import NeuroEvolution
from parl.parl_sac import PaRL_SAC
from parl.parl_td3 import PaRL_TD3

from ray.rllib.algorithms import Algorithm
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
    LAST_TARGET_UPDATE_TS,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_AGENT_STEPS_TRAINED,
    NUM_ENV_STEPS_TRAINED,
    NUM_TARGET_UPDATES,
    SYNCH_WORKER_WEIGHTS_TIMER,
    TARGET_NET_UPDATE_TIMER,
)
from parl.rollout import synchronous_parallel_sample, flatten_batches


from parl.parl import (
    evolver_algo,
    PaRL
)
from ray.rllib.utils import merge_dicts

from ray.rllib.utils.typing import (
    ResultDict,
    AlgorithmConfigDict,
    PartialAlgorithmConfigDict
)


logger = logging.getLogger(__name__)


class PaRL_PureEA:
    def setup(self, config: PartialAlgorithmConfigDict):
        super(PaRL, self).setup(config)

        self.pop_size = self.config["pop_size"]
        self.pop_config = merge_dicts(self.config, config["pop_config"])
        self.pop_config["is_pop_worker"] = True
        self.pop_workers = WorkerSet(
            env_creator=self.env_creator,
            validate_env=self.validate_env,
            policy_class=self.get_default_policy_class(self.pop_config),
            trainer_config=self.pop_config,
            num_workers=self.pop_size,
            local_worker=False,
            logdir=self.logdir,
        )
        if self.pop_size > 0:
            self.ea_config = self.config["ea_config"]
            evolver_cls = evolver_algo[self.config.get("evolver_algo", "cem")]
            self.evolver: NeuroEvolution = evolver_cls(
                self.ea_config, self.pop_workers, self.workers.local_worker())

    def training_step(self) -> ResultDict:
        train_batch_size = self.config["train_batch_size"]
        local_worker = self.workers.local_worker()

        # Step 1: Sample episodes from workers.
        target_sample_batches, pop_sample_batches = synchronous_parallel_sample(
            target_worker_set=None,
            pop_worker_set=self.pop_workers,
            episodes_per_worker=self.config["episodes_per_worker"]
        )

        assert len(target_sample_batches) == 0

        # Step 2: Store samples into replay buffer
        sample_batches = flatten_batches(target_sample_batches) + \
            flatten_batches(pop_sample_batches)

        # ts = 0  # total sample steps in the iteration
        for batch in sample_batches:
            # Update sampling step counters.
            self._counters[NUM_ENV_STEPS_SAMPLED] += batch.env_steps()
            self._counters[NUM_AGENT_STEPS_SAMPLED] += batch.agent_steps()
            # ts += batch.env_steps()
            # Store new samples in the replay buffer
            # Use deprecated add_batch() to support old replay buffers for now
            # self.local_replay_buffer.add(batch)

        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }

        # step 3: sample batches from replay buffer and place them on learner queue
        # num_train_batches = round(ts/train_batch_size*5)
        # num_train_batches = 1000
        # num_train_batches = round(ts/10)
        # for _ in range(num_train_batches):
        #     logger.info(f"add {num_train_batches} batches to learner thread")
        #     train_batch = self.local_replay_buffer.sample(train_batch_size)

        #     # replay buffer learning start size not meet
        #     if train_batch is None or len(train_batch) == 0:
        #         self.workers.local_worker().set_global_vars(global_vars)
        #         return {}

        #     # target agent is updated at background thread
        #     self._learner_thread.inqueue.put(train_batch, block=True)
        #     self._counters[NUM_SAMPLES_ADDED_TO_QUEUE] += (
        #         batch.agent_steps() if self._by_agent_steps else batch.count
        #     )

        # step 4: apply NE
        if self.pop_size > 0:
            fitnesses = self._calc_fitness(pop_sample_batches)
            # target_fitness = np.mean([episode[SampleBatch.REWARDS].sum(
            # ) for episode in flatten_batches(target_sample_batches)])

            self.evolver.evolve(fitnesses, target_fitness=None)
            # with self._timers[SYNCH_POP_WORKER_WEIGHTS_TIMER]:
            #     # set pop workers with new generated indv weights
            #     self.evolver.sync_pop_weights()

            # NEW:
            self.evolver.set_pop_weights(
                local_worker
            )

        # Update replay buffer priorities.
        # update_priorities_in_replay_buffer(
        #     self.local_replay_buffer,
        #     self.config,
        #     train_batch,
        #     train_results,
        # )

        # step 5: retrieve train_results from learner thread and update target network
        # train_results = self._process_trained_results()
        train_results = {}
        if self.pop_size > 0:
            train_results.update({
                "ea_results": self.evolver.get_iteration_results()
            })

        # step 6: sync target agent weights to rollout workers
        # Update weights and global_vars - after learning on the local worker - on all
        # remote workers.
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            self.workers.sync_weights(global_vars=global_vars)

        # Return all collected metrics for the iteration.
        return train_results
    
    
    
    def _compile_iteration_results(self, *args, **kwargs):
        result = super(PaRL, self)._compile_iteration_results(*args, **kwargs)
        
        if self.pop_size > 0:
            result["info"].update(
                self.evolver.stats()
            )
        return result


class PaRL_SAC_PureEA(PaRL_PureEA, PaRL_SAC):
    pass

class PaRL_TD3_PureEA(PaRL_PureEA, PaRL_TD3):
    pass