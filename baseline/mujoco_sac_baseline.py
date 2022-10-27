import os
import pickle
import time
from ruamel.yaml import YAML
import ray
import torch

from ray.tune import Tuner, TuneConfig
from ray.air import RunConfig, CheckpointConfig
from parl.sac import SAC_Parallel, SACConfigMod
from parl.policy import SACPolicy, SACPolicy_FixedAlpha
from parl.env_config import mujoco_config

from ray.rllib.utils.exploration import StochasticSampling
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms import Algorithm

import argparse
from dataclasses import dataclass
from typing import Union


class SAC_FixAlpha(SAC_Parallel):
    def get_default_policy_class(
            self, config):
        return SACPolicy_FixedAlpha


class SAC_TuneAlpha(SAC_Parallel):
    def get_default_policy_class(
            self, config):
        return SACPolicy


@dataclass
class Config:
    env: str = "HalfCheetah-v3"

    num_rollout_workers: int = 0
    num_eval_workers: int = 16

    rollout_fragment_length: int = 50
    enable_multiple_updates: bool = True
    rollout_vs_train: Union[int, float] = 1

    num_cpus_for_local_worker: int = 1
    num_cpus_for_rollout_worker: int = 1

    use_gpu: bool = True
    autotune_alpha: bool = False
    initial_alpha: float = 1.0
    grad_clip: float = None

    num_tests: int = 3
    training_iteration: int = 3000
    checkpoint_freq: int = 10
    evaluation_interval: int = 10

    random_timesteps: int = 0

    save_folder: str = "results"

    def resources(self):
        num_cpus = (self.num_rollout_workers*self.num_cpus_for_rollout_worker +
                    self.num_cpus_for_local_worker+self.num_eval_workers)*self.num_tests
        num_gpus = 1 if self.use_gpu else 0
        return num_cpus, num_gpus


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

    sac_config = SACConfigMod().framework('torch')
    sac_config = sac_config.rollouts(
        num_rollout_workers=config.num_rollout_workers,
        num_envs_per_worker=1,
        rollout_fragment_length=config.rollout_fragment_length,
        # no_done_at_end=True,
        horizon=1000,
        soft_horizon=False,
    )
    sac_config = sac_config.training(
        # grad_clip=config.grad_clip,
        initial_alpha=config.initial_alpha,
        train_batch_size=256,
        training_intensity=256//config.rollout_vs_train if config.enable_multiple_updates else None,
        replay_buffer_config={
            "type": "MultiAgentReplayBuffer",
            "capacity": int(1e6),
            # How many steps of the model to sample before learning starts.
            "learning_starts": 10000,
        },
        optimization={
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "entropy_learning_rate": 3e-4,
            # "actor_grad_clip": 10,
            # "critic_grad_clip": 40,
            # "alpha_grad_clip": 5
        }
    )
    sac_config = sac_config.resources(
        num_gpus=1/config.num_tests if config.use_gpu else 0,
        num_cpus_for_local_worker=config.num_cpus_for_local_worker,
        num_cpus_per_worker=config.num_cpus_for_rollout_worker
    )
    sac_config = sac_config.evaluation(
        evaluation_interval=config.evaluation_interval,
        evaluation_num_workers=config.num_eval_workers,
        evaluation_duration=10,
        evaluation_config={
            "num_envs_per_worker": 1,
            "explore": False  # greedy eval
        }
    )
    sac_config = sac_config.exploration(
        exploration_config={
            "type": StochasticSampling,
            "random_timesteps": config.random_timesteps
        }
    )
    sac_config = sac_config.reporting(
        min_time_s_per_iteration=0,
        min_sample_timesteps_per_iteration=1000,  # 1000 updates per iteration
        metrics_num_episodes_for_smoothing=5
    )

    sac_config = sac_config.environment(
        env=config.env,
        # env_config=mujoco_config.get(
        #     config.env.split("-")[0], {}).get("Parameterizable-v3", {})
    )
    sac_config = sac_config.callbacks(CPUInitCallback)
    # sac_config = sac_config.python_environment(
    #     extra_python_environs_for_driver={"OMP_NUM_THREADS": str(config.num_cpus_for_local_worker)},
    #     extra_python_environs_for_worker={"OMP_NUM_THREADS": str(config.num_cpus_for_rollout_worker)}
    # )
    sac_config = sac_config.to_dict()

    num_cpus, num_gpus = config.resources()

    ray.init(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        local_mode=False,
        include_dashboard=True
    )

    tuner = Tuner(
        SAC_TuneAlpha if config.autotune_alpha else SAC_FixAlpha,
        param_space=sac_config,
        tune_config=TuneConfig(
            num_samples=config.num_tests
        ),
        run_config=RunConfig(
            local_dir="~/ray_results",
            # this will results in 1e6 updates
            stop={"training_iteration": config.training_iteration},
            checkpoint_config=CheckpointConfig(
                num_to_keep=None,  # save all checkpoints
                checkpoint_frequency=config.checkpoint_freq
            )
        )
    )

    result_grid = tuner.fit()

    exp_name = os.path.basename(tuner._local_tuner._experiment_checkpoint_dir)
    with open(os.path.join(config.save_folder, exp_name), "wb") as f:
        pickle.dump(result_grid, f)

    time.sleep(20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str,
                        default="baseline/td3_baseline.yaml")
    parser.add_argument("--env", type=str, default=None)
    args = parser.parse_args()

    yaml = YAML(typ='safe')
    with open(args.config_file, 'r') as f:
        config = yaml.load(f)

    if args.env:
        config["env"] = args.env
    main(config)
