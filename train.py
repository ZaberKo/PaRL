from ruamel.yaml import YAML
import argparse

import ray

from ray.tune import Tuner,TuneConfig
from ray.air import RunConfig,CheckpointConfig

from parl import PaRL
from parl_config import PaRLConfig

from ray.rllib.utils import merge_dicts

def main(config):
    ray.init(num_cpus=32, num_gpus=1,local_mode=False,include_dashboard=True)


    tune_config=TuneConfig(
        num_samples=1
    )

    run_config=RunConfig(
        stop=config["stopper"],
        checkpoint_config=CheckpointConfig(
            num_to_keep=None, # save all checkpoints
            checkpoint_frequency=config["checkpoint_freq"]
        )
    )

    result_grid=Tuner(
        PaRL,
        param_space=config,
        tune_config=tune_config,
        run_config=run_config
    ).fit()

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="parl.yml")
    args = parser.parse_args()

    yaml=YAML(typ='safe')
    with open(args.config_file, 'r') as f:
        config = yaml.load(f)
    # config=namedtuple('Config',config)(**config)
    main(config)