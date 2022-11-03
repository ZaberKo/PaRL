#%%
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

import orjson

from pathlib import Path


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

def load_json_line(path):
    x=[]
    data=[]
    with path.open('r') as f:
        for line in f:
            res=orjson.loads(line)
            if "evaluation" in res:
                x.append(res["timesteps_total"])
                data.append(res["evaluation"]["hist_stats"]["episode_reward"])
    return x, data


def plot_all(path, r=0):
    plt.figure(dpi=300, figsize=(9, 9))
    eval_return_arr = []
    for exp_path in Path(path).expanduser().iterdir():
        if not exp_path.is_dir():
            continue

        print(exp_path)
        x, results=load_json_line(exp_path / "result.json")
        results=np.array(results)
        eval_return = results.mean(axis=-1)
        eval_std=results.std(axis=-1)
        smoothed_return = smooth(eval_return, r)
        smoothed_std = smooth(eval_std, r)

        eval_return_arr.append(eval_return)
        plt.plot(x, smoothed_return, lw=0.4)
        # plt.fill_between(df["Epoch"],smooth(df["MinTestEpRet"],r),smooth(df["MaxTestEpRet"],r),alpha=0.2)
        plt.fill_between(x, smoothed_return-smoothed_std,
                         smoothed_return+smoothed_std, alpha=0.2)
    _min_len = min([len(d) for d in eval_return_arr])
    avg_return = np.array([d[:_min_len] for d in eval_return_arr]).mean(axis=0)
    smoothed_avg_return = smooth(avg_return, r)
    # plt.plot(np.arange(1, len(smoothed_avg_return)+1), smoothed_avg_return)
    # plt.legend()
    plt.show()

    plt.figure(dpi=300, figsize=(9, 9))
    plt.plot(x[:_min_len], smoothed_avg_return)
    plt.show()


#%%
path=Path("~/ray_results/SAC_TuneAlpha_2022-10-28_17-15-04/SAC_TuneAlpha_Hopper-v3_10849_00000_0_2022-10-28_17-15-04/result.json").expanduser()


results=load_json_line(path)


    

# %%"
path=Path("~/ray_results")/"SAC_TuneAlpha_2022-10-28_17-15-04"
plot_all(path,r=10)
# %%

path=Path("~/ray_results")/"SAC_TuneAlpha_2022-10-28_16-58-55"
plot_all(path,r=10)

# %%
path=Path("~/ray_results")/"SAC_FixAlpha_2022-10-29_03-39-46"
plot_all(path,r=2)
# %%
