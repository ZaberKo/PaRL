import time
import os
import cloudpickle
from ruamel.yaml import YAML
import argparse
import torch

import ray
from ray.rllib.utils import merge_dicts
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms import Algorithm
from ray.tune import Tuner, TuneConfig
from ray.air import RunConfig, CheckpointConfig

from parl import PaRL_SAC_HybridES, PaRLSACConfig
from parl.env_config import mujoco_config
from parl.utils import CPUInitCallback

from tqdm import trange
from ray.rllib.utils.debug import summarize


def main(config, debug=False):
    tuner_config = config.pop("tuner_config")

    num_samples = tuner_config.get("num_samples", 1)
    if debug:
        num_samples = 1

    default_config = PaRLSACConfig()
    # default_config = default_config.callbacks(CPUInitCallback)
    # default_config = default_config.python_environment(
    #     extra_python_environs_for_driver={"OMP_NUM_THREADS": str(8)},
    #     extra_python_environs_for_worker={"OMP_NUM_THREADS": str(8)}
    # )

    # mujoco_env_setting
    env: str = config["env"]
    # env_config = mujoco_config.get(
    #     env.split("-")[0], {}).get("Parameterizable-v3", {})
    default_config = default_config.environment(
        env=env,
        # env_config=env_config
    )

    default_config = default_config.to_dict()
    merged_config = merge_dicts(default_config, config)
    merged_config["evolver_algo"] = "hybrid-es"

    trainer_resources = PaRL_SAC_HybridES.default_resource_request(
        merged_config).required_resources

    ray.init(
        num_cpus=int(trainer_resources["CPU"]*num_samples),
        num_gpus=1,
        local_mode=False,
        include_dashboard=True
    )

    if debug:
        merged_config["log_level"] = "DEBUG"
        trainer = PaRL_SAC_HybridES(config=merged_config)

        # policy=trainer.get_policy()
        # state_dict=policy.get_evolution_weights()

        for i in trange(10000):
            res = trainer.train()
            print(f"======= iter {i+1} ===========")
            del res["config"]
            del res["hist_stats"]
            print(summarize(res))
            print("+"*20)

    else:
        tune_config = TuneConfig(
            num_samples=num_samples
        )

        run_config = RunConfig(
            stop=tuner_config["stopper"],
            checkpoint_config=CheckpointConfig(
                num_to_keep=None,  # save all checkpoints
                checkpoint_frequency=tuner_config["checkpoint_freq"]
            )
        )

        tuner = Tuner(
            PaRL_SAC_HybridES,
            param_space=merged_config,
            tune_config=tune_config,
            run_config=run_config
        )

        result_grid = tuner.fit()
        exp_name = os.path.basename(
            tuner._local_tuner._experiment_checkpoint_dir)
        with open(os.path.join("results", exp_name), "wb") as f:
            cloudpickle.dump(result_grid, f)

    time.sleep(20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="PaRL-es.yaml")
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--evolver_algo", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    yaml = YAML(typ='safe')
    with open(args.config_file, 'r') as f:
        config = yaml.load(f)
    # config=namedtuple('Config',config)(**config)
    if args.env:
        config["env"] = args.env
    if args.evolver_algo:
        config["evolver_algo"] = args.evolver_algo
    main(config, args.debug)
