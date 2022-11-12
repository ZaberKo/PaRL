import os
import pickle
import time
from ruamel.yaml import YAML
import ray
import torch

from ray.tune import Tuner, TuneConfig
from ray.air import RunConfig, CheckpointConfig
from ray.rllib.algorithms.td3 import TD3Config, TD3
from ray.rllib.algorithms.ddpg.ddpg_torch_policy import DDPGTorchPolicy

from parl.env import mujoco_config
from parl.policy.td3_policy import record_grads

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms import Algorithm

from baseline.mujoco_td3_baseline import Config

import argparse
from dataclasses import dataclass

class TD3Policy(DDPGTorchPolicy):
    def extra_grad_process(self, optimizer, loss) :
        # Clip grads if configured.
        return record_grads(self, optimizer, loss)

class TD3Ori(TD3):
    def get_default_policy_class(
            self, config):
        return TD3Policy





def main(_config):
    config = Config(**_config)

    class CPUInitCallback(DefaultCallbacks):
        def on_algorithm_init(self, *, algorithm: Algorithm, **kwargs) -> None:
            # ============ driver worker multi-thread ==========
            # os.environ["OMP_NUM_THREADS"]=str(num_cpus_for_local_worker)
            # os.environ["OPENBLAS_NUM_THREADS"] = str(num_cpus_for_local_worker)
            # os.environ["MKL_NUM_THREADS"] = str(num_cpus_for_local_worker)
            # os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_cpus_for_local_worker)
            # os.environ["NUMEXPR_NUM_THREADS"] = str(num_cpus_for_local_worker)
            torch.set_num_threads(config.num_cpus_for_local_worker)

            # ============ rollout worker multi-thread ==========
            def set_rollout_num_threads(worker):
                torch.set_num_threads(config.num_cpus_for_rollout_worker)

            pendings = [w.apply.remote(set_rollout_num_threads)
                        for w in algorithm.workers.remote_workers()]
            ray.wait(pendings, num_returns=len(pendings))

    td3_config = TD3Config().framework('torch')
    td3_config = td3_config.rollouts(
        num_rollout_workers=config.num_rollout_workers,
        num_envs_per_worker=1,
        rollout_fragment_length=config.rollout_fragment_length,
    )
    td3_config = td3_config.training(
        replay_buffer_config={
            "type": "MultiAgentReplayBuffer",
            "learning_starts": 10000,
        }
    )
    td3_config = td3_config.resources(
        num_gpus=1/config.num_tests if config.use_gpu else 0,
        num_cpus_for_local_worker=config.num_cpus_for_local_worker,
        num_cpus_per_worker=config.num_cpus_for_rollout_worker
    )
    td3_config = td3_config.evaluation(
        evaluation_interval=config.evaluation_interval,
        evaluation_num_workers=config.num_eval_workers,
        evaluation_duration=10,
        evaluation_config={
            "horizon": None,
            "num_envs_per_worker": 1,
            "explore": False  # greedy eval
        }
    )
    td3_config = td3_config.exploration(
        exploration_config={
            "random_timesteps": 10000,
        }
    )
    td3_config = td3_config.reporting(
        min_time_s_per_iteration=0,
        min_sample_timesteps_per_iteration=1000,  # 1000 updates per iteration
        metrics_num_episodes_for_smoothing=5
    )

    td3_config = td3_config.environment(
        env=config.env,
        normalize_actions=False,
        env_config=mujoco_config.get(
            config.env.split("-")[0], {}).get("Parameterizable-v3", {})
    )
    td3_config = td3_config.callbacks(CPUInitCallback)
    # sac_config = sac_config.python_environment(
    #     extra_python_environs_for_driver={"OMP_NUM_THREADS": str(config.num_cpus_for_local_worker)},
    #     extra_python_environs_for_worker={"OMP_NUM_THREADS": str(config.num_cpus_for_rollout_worker)}
    # )
    td3_config = td3_config.to_dict()

    num_cpus, num_gpus = config.resources()

    ray.init(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        local_mode=False,
        include_dashboard=True
    )

    trainer=TD3Ori(config=td3_config)

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

    yaml = YAML(typ='safe')
    with open(args.config_file, 'r') as f:
        config = yaml.load(f)

    if args.env:
        config["env"]=args.env
    main(config)
