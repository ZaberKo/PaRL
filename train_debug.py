from ruamel.yaml import YAML
import argparse

import ray

from ray.tune import Tuner,TuneConfig
from ray.air import RunConfig,CheckpointConfig

from parl import PaRL, PaRLConfig


from ray.rllib.utils import merge_dicts
from ray.rllib.utils.debug import summarize
from tqdm import trange

import time
import torch


def main(config):
    ray.init(num_cpus=(8+8+16+1)*1, num_gpus=1,local_mode=False,include_dashboard=True)

    torch.set_num_threads(8)

    config.pop("tuner_config")

    config["log_level"]="DEBUG"

    trainer=PaRL(config=config)

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
    parser.add_argument("--config_file", type=str, default="PaRL.yaml")
    args = parser.parse_args()

    yaml=YAML(typ='safe')
    with open(args.config_file, 'r') as f:
        config = yaml.load(f)
    # config=namedtuple('Config',config)(**config)
    main(config)