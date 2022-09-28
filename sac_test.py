#%%
import ray
from ray.rllib.algorithms.sac import SAC, SACConfig

from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel

ray.init(num_cpus=1,local_mode=True,include_dashboard=False)

#%%
trainer:SAC = SACConfig().framework('torch') \
    .rollouts(num_rollout_workers=0, num_envs_per_worker=1,no_done_at_end=True,horizon=1000,soft_horizon=False)\
    .training(
        replay_buffer_config={
        "_enable_replay_buffer_api": True,
        "type": "MultiAgentReplayBuffer",
        "capacity": int(1e6),
        # How many steps of the model to sample before learning starts.
        "learning_starts": 10000,
    })\
    .reporting(min_time_s_per_iteration=None,min_sample_timesteps_per_iteration=1000)\
    .build(env="HalfCheetah-v3")

trainer.train()
#%%

local_worker=trainer.workers.local_worker()
policy=local_worker.get_policy()
model:SACTorchModel=policy.model
# %%
model.action_model
# %%
