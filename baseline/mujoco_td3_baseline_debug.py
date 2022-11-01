import os
import pickle
import time
from ruamel.yaml import YAML
import ray
import torch


from ray.rllib.algorithms.td3 import TD3Config
from parl.td3 import TD3Mod
from parl.env_config import mujoco_config


from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms import Algorithm

import argparse
from dataclasses import dataclass

from baseline.mujoco_td3_baseline import Config, generate_algo_config

def main(_config):
    config = Config(**_config)

    td3_config = generate_algo_config(config)

    num_cpus, num_gpus = config.resources()

    ray.init(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        local_mode=False,
        include_dashboard=False
    )

    trainer=TD3Mod(config=td3_config)

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
                        default="baseline/td3_baseline_cpu.yaml")
    parser.add_argument("--env", type=str, default=None)
    args = parser.parse_args()

    import os
    print(os.getcwd())

    yaml = YAML(typ='safe')
    with open(args.config_file, 'r') as f:
        config = yaml.load(f)

    if args.env:
        config["env"] = args.env
    main(config)
