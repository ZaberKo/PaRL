import os
import pickle
import time
from ruamel.yaml import YAML
import ray
import torch

from parl.sac import SAC_Parallel, SACConfigMod
from parl.env_config import mujoco_config

from ray.rllib.utils.exploration import StochasticSampling
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms import Algorithm

from baseline.mujoco_sac_baseline import Config, generate_algo_config

import argparse
from dataclasses import dataclass
from typing import Union


def main(_config):
    config = Config(**_config)

    sac_config = generate_algo_config(config)

    num_cpus, num_gpus = config.resources()

    ray.init(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        local_mode=False,
        include_dashboard=False
    )

    trainer = SAC_Parallel(config=sac_config)

    from tqdm import trange
    from ray.rllib.utils.debug import summarize
    for i in trange(10000):
        res = trainer.train()
        print(f"======= iter {i+1} ===========")
        del res["config"]
        del res["hist_stats"]
        print(summarize(res))
        print("+"*20)

    time.sleep(20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str,
                        default="baseline/sac_baseline_cpu.yaml")
    parser.add_argument("--env", type=str, default=None)
    args = parser.parse_args()

    yaml = YAML(typ='safe')
    with open(args.config_file, 'r') as f:
        config = yaml.load(f)

    if args.env:
        config["env"] = args.env
    main(config)
