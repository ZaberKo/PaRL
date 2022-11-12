import numpy as np
import gym
from gym.wrappers import TimeLimit
from ray.tune.registry import register_env

mujoco_config = {
    'Swimmer': {  # 2 DoF
    },
    'Hopper': {  # 3 DoF
    },
    'HalfCheetah': {  # 6 DoF
    },
    'Walker2d': {  # 6 DoF
    },
    'Ant': {  # 8 DoF
        'Parameterizable-v3': {
            'healthy_reward': 0.0,
            'healthy_z_range': (-np.inf, np.inf),
            'exclude_current_positions_from_observation': False,
        }
    },
    'Humanoid': {  # 17 DoF
        'Parameterizable-v3': {
            'healthy_reward': 0.0,
            'healthy_z_range': (-np.inf, np.inf),
            'exclude_current_positions_from_observation': False,
        }
    },
}

class NoDoneAtTimeLimit(gym.Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if info.get("TimeLimit.truncated", False):
            done = False            
        return obs, reward, done, info


def register_my_env(env_name, max_episode_steps: int = None):
    def env_creator(env_config):
        env = gym.make(env_name, **env_config)
        env = NoDoneAtTimeLimit(env)
        return env

    new_env_name = env_name # use the original name?
    register_env(new_env_name, env_creator)

    return new_env_name