#%%

import gym
from gym import envs

env=gym.make("Swimmer-v3")
# %%
env._max_episode_steps
# %%
env.reset()
rewards =[]
while True:
    action=env.action_space.sample()
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
    if done:
        break
# %%
for _ in range(10):
    print(env.step(action))
# %%
