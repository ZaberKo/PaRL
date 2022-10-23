#%%
import torch
import torch.nn as nn
from ray.rllib.algorithms.ddpg.ddpg_torch_model import DDPGTorchModel

import gym

env=gym.make("HalfCheetah-v3")



# #%%
# model=DDPGTorchModel(
#     obs_space=env.observation_space,
#     action_space=env.action_space,
#     num_outputs=env.action_space.shape[0],
#     model_config={},
#     name="TestModel"
# )
# # %%
# model
# # %%
# model({"obs": env.observation_space.sample()})
# %%
from ray.rllib.algorithms.ddpg import DDPG,DDPGConfig

config=DDPGConfig().framework('torch').environment(env="HalfCheetah-v3").to_dict()
ddpg=DDPG(config=config)

policy=ddpg.get_policy()
# %%
policy.model
# %%
policy.model({"obs": torch.as_tensor(env.observation_space.sample())})
# %%
policy.model
# %%
import inspect

print(inspect.getsource(policy.model.forward))
# %%
