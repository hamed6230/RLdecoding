# ============================================================== #
# imports {{{
# ============================================================== #
import sys
import ray
import ray.rllib.algorithms.apex_dqn.apex_dqn as dqn
from ray.rllib.algorithms.apex_dqn.apex_dqn import ApexDQNConfig
from ray.rllib.algorithms.algorithm import with_common_config
from ray import tune
from CodeEnv import *
import ctypes
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



ctypes.windll.shell32.IsUserAnAdmin()
# }}}
# ============================================================== #
# config {{{
# ============================================================== #

codedir = r"\MYP\fc94\RL_SOURCE"
tmpdir = r"\MYP\fc94"


config = ApexDQNConfig()
config = config.environment(
    env_config=dict(code="RM_3_7_std", EbNo_dB=4, maxIter=10, WBF=False, asort=True, path_to_Hmat=codedir + "/Hmat"),
    env=CodeEnv)
config = config.training(num_atoms=1, v_min=-10.0, v_max=10.0, sigma0=0.5, dueling=False, double_q=False, n_step=1,
                         model=dict(fcnet_activation="relu", fcnet_hiddens=[500]),
                         train_batch_size=32,
                         lr=0.0001,
                         lr_schedule=None,
                         adam_epsilon=1e-8,
                         grad_clip=40,
                         num_steps_sampled_before_learning_starts=1000,
                         optimizer=dict(max_weight_sync_delay=2000, num_replay_buffer_shards=5, debug=False)
                         )
config = config.resources(num_gpus=0)
config = config.rollouts(num_rollout_workers=2)
buffer_conf = config.replay_buffer_config.update({
    # "no_local_replay_buffer": True,
    # Specify prioritized replay by supplying a buffer type that supports
    "capacity": 200000,
    # Alpha parameter for prioritized replay buffer.
    "prioritized_replay_alpha": 0.6,
    # Beta parameter for sampling from prioritized replay buffer.
    "prioritized_replay_beta": 0.4,
    # Epsilon to add to the TD errors when updating priorities.
    "prioritized_replay_eps": 1e-6,

    "worker_side_prioritization": True,

})
print(config)



# }}}
# ============================================================== #
# run optimizations {{{
# ============================================================== #
# ray.init()
ray.init(num_gpus=0,ignore_reinit_error=True)

tune.run(
    "APEX",
    # "DQN",
    fail_fast=True,
    name="CodeEnv",
    checkpoint_freq=15,
    checkpoint_at_end=True,
    num_samples=1,
    local_dir=codedir + "/ray_results",
    stop={"training_iteration": 150},
    config=config
)

ray.shutdown()

print("done!")
# }}}
# ============================================================== #
