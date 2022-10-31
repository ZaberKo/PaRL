# %%
import cloudpickle
import ray
import gym
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from ray.rllib.evaluation import SampleBatch

from parl.sac import SAC_Parallel

import argparse


class RolloutResult:
    def __init__(self, num_checkpoints, num_episodes) -> None:
        self.results_ret = np.zeros(shape=(num_checkpoints, num_episodes))
        self.results_len = np.zeros(shape=(num_checkpoints, num_episodes))

    def add(self, i, j, ep_ret, ep_len):
        self.results_ret[i][j] = ep_ret
        self.results_len[i][j] = ep_len


def main(args):
    path = args.path

    with (Path(path).expanduser()/"params.pkl").open('rb') as f:
        config = pickle.load(f)
    config.update({
        "num_gpus": 0,
        "num_workers": 0,
        "explore": False,
    })
    trainer = SAC_Parallel(config=config)

    if args.env:
        config['env'] = args.env

    print(config['env'])
    env = gym.make(config['env'])
    num_episodes = args.num_episodes

    policy = trainer.get_policy()

    results = evaluate(path, env, policy, num_episodes=num_episodes)
    plot(results)


def evaluate(path, env, policy, num_episodes=10):
    checkpoint_folders = sorted(Path(path).expanduser().glob("checkpoint*"))
    # print(checkpoint_folders)

    results = RolloutResult(len(checkpoint_folders), num_episodes)

    for i, checkpoint_folder in enumerate(checkpoint_folders):
        checkpoint = next(checkpoint_folder.glob("checkpoint*"))
        with checkpoint.open('rb') as f:
            state_dict = pickle.load(f)
        worker_state = pickle.loads(state_dict['worker'])
        policy_state_dict = worker_state["state"]["default_policy"]
        policy.set_state(policy_state_dict)

        for j in range(num_episodes):
            o, d, ep_ret, ep_len = env.reset(), False, 0, 0
            while not d:
                a, _, a_info = policy.compute_actions_from_input_dict(
                    SampleBatch({'obs': o}), explore=False)
                o, r, d, _ = env.step(a)
                ep_ret += r
                ep_len += 1

            results.add(i, j, ep_ret, ep_len)
        print(
            f"{checkpoint_folder.name}: {np.mean(results.results_ret[i]):.2f} ± {np.std(results.results_ret[i]):.2f}")
        env.reset()

    return results


def plot(results: RolloutResult, r=0):
    plt.figure(dpi=300, figsize=(6, 3))

    avg_return = np.mean(results.results_ret, axis=-1)
    std_return = np.std(results.results_ret, axis=-1)

    smoothed_return = smooth(avg_return, r)
    smoothed_std = smooth(std_return, r)

    checkpoints = len(avg_return)
    timesteps = np.arange(10000, 10000*(checkpoints+1), 10000)
    plt.plot(timesteps, smoothed_return, lw=0.4)
    plt.fill_between(timesteps, smoothed_return-smoothed_std,
                     smoothed_return+smoothed_std, alpha=0.2)
    plt.show()


def smooth(y, radius, mode='two_sided', valid_only=False):
    '''Smooth signal y, where radius is determines the size of the window.

    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]
    valid_only: put nan in entries where the full-sized window is not available
    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2 * radius + 1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius + 1)
        out = np.convolve(y, convkernel, mode='same') / \
            np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel, mode='full') / \
            np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius + 1]
        if valid_only:
            out[:radius] = np.nan
    return out


# %%
parser = argparse.ArgumentParser()
parser.add_argument(
    "--path", type=str, default="~/ray_results/SAC_Parallel_2022-10-31_09-35-33/SAC_Parallel_Hopper-v3_4fa66_00000_0_2022-10-31_09-35-33")
parser.add_argument("--env", type=str, default=None)
parser.add_argument("--num_episodes", type=int, default=10)
args = parser.parse_args(args=[])
main(args)

# %%
