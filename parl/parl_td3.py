from parl.parl import PaRL, PaRLBaseConfig

from parl.policy import TD3Policy
from parl.td3 import TD3ConfigMod
from ray.rllib.algorithms.td3 import TD3

from ray.rllib.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import (
    AlgorithmConfigDict,
    PartialAlgorithmConfigDict
)
from typing import Optional, Type


class PaRLTD3Config(PaRLBaseConfig, TD3ConfigMod):
    def __init__(self, algo_class=None):
        TD3ConfigMod.__init__(self, algo_class=algo_class or PaRL_TD3)

        self.add_actor_layer_norm = True
        self.training(
            # grad_clip=config.grad_clip,
            critic_lr=3e-4,
            actor_lr=3e-4,
            tau=5e-3,
            policy_delay=2,
            target_noise=0.2,
            target_noise_clip=0.5,
            smooth_target_policy=True,
            train_batch_size=256,
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": int(1e6),
                # How many steps of the model to sample before learning starts.
                "learning_starts": 1000,
            },
            actor_hiddens=[256,256],
            critic_hiddens=[256,256]
        )
        self.exploration(
            exploration_config={
                # TD3 uses simple Gaussian noise on top of deterministic NN-output
                # actions (after a possible pure random phase of n timesteps).
                "type": "GaussianNoise",
                # For how many timesteps should we return completely random
                # actions, before we start adding (scaled) noise?
                "random_timesteps": 0,
                # Gaussian stddev of action noise for exploration.
                "stddev": 0.1,
                # Scaling settings by which the Gaussian noise is scaled before
                # being added to the actions. NOTE: The scale timesteps start only
                # after(!) any random steps have been finished.
                # By default, do not anneal over time (fixed 1.0).
                "initial_scale": 1.0,
                "final_scale": 1.0,
                "scale_timesteps": 1,
            }
        )

        PaRLBaseConfig.__init__(self)


class PaRL_TD3(PaRL, TD3):
    @classmethod
    @override(TD3)
    def get_default_config(cls) -> AlgorithmConfigDict:
        return PaRLTD3Config().to_dict()

    @override(TD3)
    def get_default_policy_class(
        self, config: PartialAlgorithmConfigDict
    ) -> Optional[Type[Policy]]:
        return TD3Policy
