# %%

import gym
import ray
from ray.rllib.algorithms.sac import SACConfig
from sac import SAC_Parallel

from ray.tune import Tuner, TuneConfig
from ray.air import RunConfig, CheckpointConfig

from policy import SACPolicy,SACPolicy_FixedAlpha


from ray.rllib.utils.debug import summarize
from tqdm import trange



num_rollout_workers=0
num_envs_per_worker=4

rollout_vs_train=1

config = SACConfig().framework('torch') \
    .rollouts(
        # rollout_fragment_length=1, # already set in SAC
        num_rollout_workers=num_rollout_workers,
        num_envs_per_worker=num_envs_per_worker,
        no_done_at_end=True,
        horizon=1000,
        soft_horizon=False)\
    .training(
        grad_clip=40,
        initial_alpha=1,
        train_batch_size=256,
        training_intensity=256/rollout_vs_train,
        replay_buffer_config={
        "_enable_replay_buffer_api": True,
        "type": "MultiAgentReplayBuffer",
        "capacity": int(1e6),
        # How many steps of the model to sample before learning starts.
        "learning_starts": 1000,
    })\
    .resources(num_gpus=0.1)\
    .evaluation(
        evaluation_interval=10, 
        evaluation_num_workers=1, 
        evaluation_duration=10,
        evaluation_config={
            "no_done_at_end":False,
            "horizon":None,
            "num_envs_per_worker": 1
    })\
    .reporting(
        min_time_s_per_iteration=0,
        min_sample_timesteps_per_iteration=1000, # 1000 updates per iteration
        metrics_num_episodes_for_smoothing=5
        ) \
    .environment(env="HalfCheetah-v3")\
    .to_dict()

#%%
print(summarize(config))

#%%
class SAC_FixAlpha_Parallel(SAC_Parallel):
    def get_default_policy_class(self, config):
        return SACPolicy_FixedAlpha
#%%
# trainer=SAC_FixAlpha_Parallel(config=config)

#%%
# calculate_rr_weights(config)
#%%

ray.init(num_cpus=(num_rollout_workers+1+1), num_gpus=1, local_mode=False, include_dashboard=False)
trainer=SAC_FixAlpha_Parallel(config=config)

#%%
# policy=trainer.get_policy()
# model=policy.model

#%%


#%%
for i in trange(1000):
    res = trainer.train()
    print(f"======= iter {i+1} ===========")
    del res["config"]
    del res["hist_stats"]
    print(summarize(res))
    print("+"*20)


import time
time.sleep(10)

# %%



# %%
