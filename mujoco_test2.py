# %%
import gym
import ray
from ray.rllib.algorithms.sac import SAC, SACConfig

from ray.tune import Tuner, TuneConfig
from ray.air import RunConfig, CheckpointConfig

from policy import SACPolicy

ray.init(num_cpus=17*3, num_gpus=1, local_mode=False, include_dashboard=True)

config = SACConfig().framework('torch') \
    .rollouts(num_rollout_workers=0, num_envs_per_worker=1,no_done_at_end=True,horizon=1000,soft_horizon=False)\
    .training(
        initial_alpha=1,
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
        evaluation_num_workers=16, 
        evaluation_duration=16,
        evaluation_config={
            "no_done_at_end":False,
            "horizon":None
    })\
    .reporting(
        min_time_s_per_iteration=None,
        min_sample_timesteps_per_iteration=1000, # 1000 updates per iteration
        metrics_num_episodes_for_smoothing=5
        ) \
    .environment(env="HalfCheetah-v3")\
    .to_dict()

    

class MySAC(SAC):
    def get_default_policy_class(
        self, config):
        return SACPolicy

result_grid = Tuner(
    MySAC,
    param_space=config,
    tune_config=TuneConfig(
        num_samples=3
    ),
    run_config=RunConfig(
        stop={"training_iteration": 1000}, # this will results in 1e6 updates
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
