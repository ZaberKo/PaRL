from ray.rllib.algorithms.sac import SAC, SACConfig
from parl import PaRL


class PaRLConfig(SACConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or PaRL)

        self.q_model_config = {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
            "post_fcnet_hiddens": [],
            "post_fcnet_activation": None,
            "custom_model": None,  # Use this to define custom Q-model(s).
            "custom_model_config": {},
        }
        self.policy_model_config = {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
            "post_fcnet_hiddens": [],
            "post_fcnet_activation": None,
            "custom_model": None,  # Use this to define a custom policy model.
            "custom_model_config": {},
        }

        self.store_buffer_in_checkpoints = False
        self.replay_buffer_config = {
            "type": "MultiAgentReplayBuffer",
            "capacity": 2000000,
            "learning_starts": 50000,
            # "no_local_replay_buffer": True,
            "replay_buffer_shards_colocated_with_driver": False,
            # "num_replay_buffer_shards": 1
        }

        self.optimization = {
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "entropy_learning_rate": 3e-4,
        }

        self.episodes_per_worker=1

        # EA config
        self.pop_size=10
        self.pop_config={}
        self.elite_fraction=0.5
        self.noise_decay_coeff=0.95
        self.noise_init=1e-3
        self.noise_end=1e-5

        # training config
        self.n_step = 1
        self.initial_alpha = 1.0
        self.tau=0.005
        self.normalize_actions=True

        # learner thread config
        self.num_multi_gpu_tower_stacks= 8
        self.learner_queue_size: 16
        self.num_data_load_threads: 16

        self.target_network_update_freq=1 # unit: iteration

        # reporting
        self.metrics_episode_collection_timeout_s = 60.0
        self.metrics_num_episodes_for_smoothing = 100
        self.min_time_s_per_iteration = None
        self.min_sample_timesteps_per_iteration = 0
        self.min_train_timesteps_per_iteration = 500

        # default_resources
        self.num_cpus_per_worker=1
        self.num_envs_per_worker=1
        self.num_cpus_for_local_worker=1
        self.num_gpus_per_worker=0



        self.framework("torch")
