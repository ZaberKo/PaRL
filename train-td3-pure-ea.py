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

from parl import PaRLTD3Config, PaRL_TD3_PureEA
from parl.env import mujoco_config


def main(config):
    tuner_config = config.pop("tuner_config")

    num_samples = tuner_config.get("num_samples", 1)
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

    class CPUInitCallback(DefaultCallbacks):
        def __init__(self):
            super().__init__()
            self.num_cpus_for_local_worker = 4
            self.num_cpus_for_rollout_worker = 1

        def on_algorithm_init(self, *, algorithm: Algorithm, **kwargs) -> None:
            # ============ driver worker multi-thread ==========
            n = int(self.num_cpus_for_local_worker)
            os.environ["OMP_NUM_THREADS"] = str(n)
            os.environ["OPENBLAS_NUM_THREADS"] = str(n)
            os.environ["MKL_NUM_THREADS"] = str(n)
            os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
            os.environ["NUMEXPR_NUM_THREADS"] = str(n)
            torch.set_num_threads(n)

            # ============ rollout worker multi-thread ==========
            def set_rollout_num_threads(worker):
                torch.set_num_threads(self.num_cpus_for_rollout_worker)

            pendings = [w.apply.remote(set_rollout_num_threads)
                        for w in algorithm.workers.remote_workers()]
            ray.wait(pendings, num_returns=len(pendings))

    default_config = PaRLTD3Config()
    default_config = default_config.callbacks(CPUInitCallback)
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
    default_config = default_config.training(
        add_actor_layer_norm=True
    )

    default_config = default_config.to_dict()
    merged_config = merge_dicts(default_config, config)

    trainer_resources = PaRL_TD3_PureEA.default_resource_request(
        merged_config).required_resources

    ray.init(
        num_cpus=int(trainer_resources["CPU"]*num_samples),
        num_gpus=0,
        local_mode=False,
        include_dashboard=True
    )
    tuner = Tuner(
        PaRL_TD3_PureEA,
        param_space=merged_config,
        tune_config=tune_config,
        run_config=run_config
    )

    result_grid = tuner.fit()
    exp_name = os.path.basename(tuner._local_tuner._experiment_checkpoint_dir)
    with open(os.path.join("results", exp_name), "wb") as f:
        cloudpickle.dump(result_grid, f)

    time.sleep(60)


    # trainer=PaRL_TD3_PureEA(config=merged_config)
    # from tqdm import trange
    # from ray.rllib.utils.debug import summarize

    # for i in trange(10000):
    #     res = trainer.train()
    #     print(f"======= iter {i+1} ===========")
    #     del res["config"]
    #     del res["hist_stats"]
    #     print(summarize(res))
    #     print("+"*20)

    # time.sleep(20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="exp_config/PaRL-td3-pure-cem.yaml")
    parser.add_argument("--env", type=str, default=None)
    args = parser.parse_args()

    yaml = YAML(typ='safe')
    with open(args.config_file, 'r') as f:
        config = yaml.load(f)
    # config=namedtuple('Config',config)(**config)
    if args.env:
        config["env"] = args.env
    main(config)
