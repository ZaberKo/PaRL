# %%
import numpy as np
import gym
import ray
from ray.rllib.algorithms.sac import  SACConfig
from sac import SAC_Parallel

from ray.tune import Tuner, TuneConfig
from ray.air import RunConfig, CheckpointConfig

from policy import SACPolicy,SACPolicy_FixedAlpha


num_test=3

num_rollout_workers=0
num_envs_per_worker=1

rollout_vs_train=1

num_eval_workers=16

config = SACConfig().framework('torch') \
    .rollouts(
        rollout_fragment_length=50, # already set in SAC
        num_rollout_workers=num_rollout_workers,
        num_envs_per_worker=num_envs_per_worker,
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
        "capacity": int(1e6),
        # How many steps of the model to sample before learning starts.
        "learning_starts": 10000,
    })\
    .resources(num_gpus=0.1)\
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
    .reporting(
        min_time_s_per_iteration=0,
        min_sample_timesteps_per_iteration=1000, # 1000 updates per iteration
        metrics_num_episodes_for_smoothing=5
        ) \
    .environment(env="HalfCheetah-v3")\
    .to_dict()


class SAC_FixAlpha_Parallel(SAC_Parallel):
    def get_default_policy_class(self, config):
        return SACPolicy_FixedAlpha

#%%
# calculate_rr_weights(config)
#%%

ray.init(num_cpus=(num_rollout_workers+1+num_eval_workers)*num_test, num_gpus=1, local_mode=False, include_dashboard=True)
result_grid = Tuner(
    SAC_FixAlpha_Parallel,
    param_space=config,
    tune_config=TuneConfig(
        num_samples=num_test
    ),
    run_config=RunConfig(
        stop={"training_iteration": 3000}, # this will results in 1e6 updates
        checkpoint_config=CheckpointConfig(
            num_to_keep=None,  # save all checkpoints
            checkpoint_frequency=10
        )
    )
).fit()

import time
time.sleep(10)

# %%



# %%
