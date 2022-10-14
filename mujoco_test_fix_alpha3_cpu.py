# %%
import gym
import ray
from ray.rllib.algorithms.sac import SACConfig
from sac import SAC_Parallel

from ray.tune import Tuner, TuneConfig
from ray.air import RunConfig, CheckpointConfig

from policy import SACPolicy,SACPolicy_FixedAlpha

from ray.rllib.utils.exploration import StochasticSampling

from ray.rllib.algorithms.callbacks import DefaultCallbacks
import torch

num_tests=3
num_eval_workers=16

rollout_vs_train=1

num_cpus_for_local_worker=16
class CPUInitCallback(DefaultCallbacks):
    def on_algorithm_init(self, *, algorithm: "Algorithm", **kwargs) -> None:
        # os.environ["OMP_NUM_THREADS"]=str(num_cpus_for_local_worker)
        # os.environ["OPENBLAS_NUM_THREADS"] = str(num_cpus_for_local_worker)
        # os.environ["MKL_NUM_THREADS"] = str(num_cpus_for_local_worker)
        # os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_cpus_for_local_worker) 
        # os.environ["NUMEXPR_NUM_THREADS"] = str(num_cpus_for_local_worker)
        torch.set_num_threads(num_cpus_for_local_worker)


config = SACConfig().framework('torch') \
    .rollouts(
        # rollout_fragment_length=1, # already set in SAC
        num_rollout_workers=0,
        num_envs_per_worker=1,
        rollout_fragment_length=50,
        no_done_at_end=True,
        horizon=1000,
        soft_horizon=False)\
    .training(
        initial_alpha=0.2,
        train_batch_size=256,
        training_intensity=256/rollout_vs_train,
        replay_buffer_config={
        "_enable_replay_buffer_api": True,
        "type": "MultiAgentReplayBuffer",
        "capacity": int(1e5),
        # How many steps of the model to sample before learning starts.
        "learning_starts": 10000,
    })\
    .resources(
        num_gpus=0,
        num_cpus_for_local_worker=num_cpus_for_local_worker
    )\
    .evaluation(
        evaluation_interval=10, 
        evaluation_num_workers=num_eval_workers, 
        evaluation_duration=num_eval_workers*1,
        evaluation_config={
            "no_done_at_end":False,
            "horizon":None,
            "num_envs_per_worker": 1,
            "explore": False # greedy eval
    })\
    .exploration(
        exploration_config={
            "type": StochasticSampling,
            "random_timesteps": 10000
        }
    )\
    .reporting(
        min_time_s_per_iteration=0,
        min_sample_timesteps_per_iteration=1000, # 1000 updates per iteration
        metrics_num_episodes_for_smoothing=5
        ) \
    .environment(env="HalfCheetah-v3")\
    .callbacks(CPUInitCallback)\
    .to_dict()


#%%
# from sac import calculate_rr_weights
# calculate_rr_weights(config)

#%%
class SAC_FixAlpha(SAC_Parallel):
    def get_default_policy_class(
        self, config):
        return SACPolicy_FixedAlpha

ray.init(num_cpus=(num_eval_workers+num_cpus_for_local_worker)*num_tests, num_gpus=1, local_mode=False, include_dashboard=True)
result_grid = Tuner(
    SAC_FixAlpha,
    param_space=config,
    tune_config=TuneConfig(
        num_samples=num_tests
    ),
    run_config=RunConfig(
        local_dir="~/ray_results",
        stop={"training_iteration": 3000}, # this will results in 1e6 updates
        checkpoint_config=CheckpointConfig(
            num_to_keep=None,  # save all checkpoints
            checkpoint_frequency=100
        )
    )
).fit()

import time
time.sleep(10)

# %%



# %%
