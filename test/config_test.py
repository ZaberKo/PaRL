#%%
from ray.rllib.algorithms.td3 import TD3Config
from ray.rllib.utils.debug import summarize

config=TD3Config().to_dict()
print(summarize(config))
# %%
