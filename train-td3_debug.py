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

from tqdm import trange
from ray.rllib.utils.debug import summarize

from parl import PaRL_TD3, PaRLTD3Config
from parl.env_config import mujoco_config


class CPUInitCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.num_cpus_for_local_worker = 8
        self.num_cpus_for_rollout_worker = 8

    def on_algorithm_init(self, *, algorithm: Algorithm, **kwargs) -> None:
        # ============ driver worker multi-thread ==========
        # os.environ["OMP_NUM_THREADS"]=str(num_cpus_for_local_worker)
        # os.environ["OPENBLAS_NUM_THREADS"] = str(num_cpus_for_local_worker)
        # os.environ["MKL_NUM_THREADS"] = str(num_cpus_for_local_worker)
        # os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_cpus_for_local_worker)
        # os.environ["NUMEXPR_NUM_THREADS"] = str(num_cpus_for_local_worker)
        torch.set_num_threads(self.num_cpus_for_local_worker)

        # ============ rollout worker multi-thread ==========
        def set_rollout_num_threads(worker):
            torch.set_num_threads(self.num_cpus_for_rollout_worker)

        pendings = [w.apply.remote(set_rollout_num_threads)
                    for w in algorithm.workers.remote_workers()]
        ray.wait(pendings, num_returns=len(pendings))


def main(config):
    tuner_config = config.pop("tuner_config")

    default_config = PaRLTD3Config()
    # default_config = default_config.callbacks(CPUInitCallback)
    # default_config = default_config.python_environment(
    #     extra_python_environs_for_driver={"OMP_NUM_THREADS": str(8)},
    #     extra_python_environs_for_worker={"OMP_NUM_THREADS": str(8)}
    # )

    # mujoco_env_setting
    env: str = config["env"]
    # env_config = mujoco_config.get(
    #     env.split("-")[0], {}).get("Parameterizable-v3", {})
    default_config= default_config.environment(
        env=env, 
        # env_config=env_config
        )

    default_config = default_config.to_dict()
    merged_config = merge_dicts(default_config, config)

    trainer_resources = PaRL_TD3.default_resource_request(
        merged_config).required_resources

    ray.init(
        num_cpus=int(trainer_resources["CPU"]),
        num_gpus=1,
        local_mode=False,
        include_dashboard=True
    )
    trainer=PaRL_TD3(config=merged_config)

    # policy=trainer.get_policy()
    # state_dict=policy.get_evolution_weights()


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
    parser.add_argument("--config_file", type=str, default="PaRL-td3.yaml")
    parser.add_argument("--env", type=str, default=None)
    args = parser.parse_args()

    yaml = YAML(typ='safe')
    with open(args.config_file, 'r') as f:
        config = yaml.load(f)
    # config=namedtuple('Config',config)(**config)
    if args.env:
        config["env"]=args.env
    main(config)
