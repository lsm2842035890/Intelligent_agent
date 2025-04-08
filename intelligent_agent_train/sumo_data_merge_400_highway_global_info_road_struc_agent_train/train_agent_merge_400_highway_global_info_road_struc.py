import ray
from ray import tune
import glob
from ray.tune.registry import register_env
from train_agent_merge_400_highway_global_info_road_struc_env_sumo_data import D2RLTrainingEnv
import yaml, argparse
from tqdm import tqdm
import json
import ray.rllib.models.catalog

parser = argparse.ArgumentParser()
parser.add_argument('--yaml_conf', type=str, default=r'C:\Users\28420\Desktop\SAIC\Dense-Deep-Reinforcement-Learning-main\sumo_data_merge_400_highway_global_info_road_struc_agent_train\train_merge_400_s_d_global_info_road_struc.yaml', metavar='N',
                    help='the yaml configuration file path')
args = parser.parse_args()

try:
    with open(args.yaml_conf, 'r') as f:
        yaml_conf = yaml.load(f, Loader=yaml.FullLoader)
except Exception as e:
    print("Yaml configuration file not successfully loaded:", e)


def env_creator(env_config):
    return D2RLTrainingEnv(yaml_conf)

register_env("my_env", env_creator)
ray.init(include_dashboard=False, ignore_reinit_error=True)


import os
from typing import Dict, TYPE_CHECKING
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import MultiAgentEpisode

if TYPE_CHECKING:
    from ray.rllib.evaluation import RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        episode.hist_data["constant"] = []
        episode.hist_data["weight_reward"] = []
        episode.hist_data["exposure"] = []
        episode.hist_data["positive_weight_reward"] = []
        episode.hist_data["episode_num"] = []
        episode.hist_data["step_num"] = []

    def on_episode_end(self, *, worker: "RolloutWorker", base_env: BaseEnv, policies: Dict[str, Policy],
                       episode: MultiAgentEpisode, env_index: int, **kwargs):
        last_info = episode.last_info_for()
        # for key in episode.hist_data:
        #     episode.hist_data[key].append(last_info[key])
        # print(last_info)


print("Nodes in the Ray cluster:")
print(ray.nodes())
tune.run(
    "PPO",
    stop={"training_iteration": 1000},
    config={
        "env": "my_env",
        # "num_gpus": 0,
        "num_gpus": 0,
        "num_workers": yaml_conf["num_workers"],
        "num_envs_per_worker": 1,
        "gamma": 1.0,
        "rollout_fragment_length": 600,
        "vf_clip_param": yaml_conf["clip_reward_threshold"],
        "framework": "torch",
        "ignore_worker_failures": True,
        "callbacks": MyCallbacks,
        #######
        "model": {
            # 启用LSTM
            "use_lstm": True,
            # LSTM序列最大长度，默认20
            "max_seq_len": 20,
            # LSTM单元的大小，默认256
            "lstm_cell_size": 256,
            # 是否使用前一个动作作为LSTM的输入
            "lstm_use_prev_action": False,
            # 是否使用前一个奖励作为LSTM的输入
            "lstm_use_prev_reward": False,
            # LSTM数据格式：时间优先 (TxBx..) 或批次优先 (BxTx..)
            "_time_major": False,
        },
        #####
    },
    checkpoint_freq=100,
    local_dir=yaml_conf["local_dir"],
    name=yaml_conf["experiment_name"],
)



# {'num_workers': 2, 'num_envs_per_worker': 1, 'create_env_on_driver': False, 'rollout_fragment_length': 200, 'batch_mode': 'truncate_episodes', 'gamma': 0.99, 'lr': 5e-05, 'train_batch_size': 4000, 'model': {'_use_default_native_models': False, '_disable_preprocessor_api': False, '_disable_action_flattening': False, 'fcnet_hiddens': [256, 256], 'fcnet_activation': 'tanh', 'conv_filters': None, 'conv_activation': 'relu', 'post_fcnet_hiddens': [], 'post_fcnet_activation': 'relu', 'free_log_std': False, 'no_final_linear': False, 'vf_share_layers': False, 'use_lstm': False, 'max_seq_len': 20, 'lstm_cell_size': 256, 'lstm_use_prev_action': False, 'lstm_use_prev_reward': False, '_time_major': False, 'use_attention': False, 'attention_num_transformer_units': 1, 'attention_dim': 64, 'attention_num_heads': 1, 'attention_head_dim': 32, 'attention_memory_inference': 50, 'attention_memory_training': 50, 'attention_position_wise_mlp_dim': 32, 'attention_init_gru_gate_bias': 2.0, 'attention_use_n_prev_actions': 0, 'attention_use_n_prev_rewards': 0, 'framestack': True, 'dim': 84, 'grayscale': False, 'zero_mean': True, 'custom_model': None, 'custom_model_config': {}, 'custom_action_dist': None, 'custom_preprocessor': None, 'lstm_use_prev_action_reward': -1}, 'optimizer': {}, 'horizon': None, 'soft_horizon': False, 'no_done_at_end': False, 'env': None, 'observation_space': None, 'action_space': None, 'env_config': {}, 'remote_worker_envs': False, 'remote_env_batch_wait_ms': 0, 'env_task_fn': None, 'render_env': False, 'record_env': False, 'clip_rewards': None, 'normalize_actions': True, 'clip_actions': False, 'preprocessor_pref': 'deepmind', 'log_level': 'WARN', 'callbacks': <class 'ray.rllib.agents.callbacks.DefaultCallbacks'>, 'ignore_worker_failures': False, 'log_sys_usage': True, 'fake_sampler': False, 'framework': 'tf', 'eager_tracing': False, 'eager_max_retraces': 20, 'explore': True, 'exploration_config': {'type': 'StochasticSampling'}, 'evaluation_interval': None, 'evaluation_duration': 10, 'evaluation_duration_unit': 'episodes', 'evaluation_parallel_to_training': False, 'in_evaluation': False, 'evaluation_config': {}, 'evaluation_num_workers': 0, 'custom_eval_function': None, 'always_attach_evaluation_results': False, 'sample_async': False, 'sample_collector': <class 'ray.rllib.evaluation.collectors.simple_list_collector.SimpleListCollector'>, 'observation_filter': 'NoFilter', 'synchronize_filters': True, 'tf_session_args': {'intra_op_parallelism_threads': 2, 'inter_op_parallelism_threads': 2, 'gpu_options': {'allow_growth': True}, 'log_device_placement': False, 'device_count': {'CPU': 1}, 'allow_soft_placement': True}, 'local_tf_session_args': {'intra_op_parallelism_threads': 8, 'inter_op_parallelism_threads': 8}, 'compress_observations': False, 'metrics_episode_collection_timeout_s': 180, 'metrics_num_episodes_for_smoothing': 100, 'min_time_s_per_reporting': None, 'min_train_timesteps_per_reporting': None, 'min_sample_timesteps_per_reporting': None, 'seed': None, 'extra_python_environs_for_driver': {}, 'extra_python_environs_for_worker': {}, 'num_gpus': 0, '_fake_gpus': False, 'num_cpus_per_worker': 1, 'num_gpus_per_worker': 0, 'custom_resources_per_worker': {}, 'num_cpus_for_driver': 1, 'placement_strategy': 'PACK', 'input': 'sampler', 'input_config': 
# {}, 'actions_in_input_normalized': False, 'input_evaluation': ['is', 'wis'], 'postprocess_inputs': False, 'shuffle_buffer_size': 0, 'output': None, 'output_compress_columns': ['obs', 'new_obs'], 'output_max_file_size': 67108864, 'multiagent': {'policies': {}, 'policy_map_capacity': 100, 'policy_map_cache': None, 'policy_mapping_fn': None, 'policies_to_train': None, 'observation_fn': None, 'replay_mode': 'independent', 'count_steps_by': 'env_steps'}, 'logger_config': None, '_tf_policy_handles_more_than_one_loss': False, '_disable_preprocessor_api': False, '_disable_action_flattening': False, '_disable_execution_plan_api': False, 'simple_optimizer': -1, 'monitor': -1, 'evaluation_num_episodes': -1, 'metrics_smoothing_episodes': -1, 'timesteps_per_iteration': 0, 'min_iter_time_s': -1, 'collect_metrics_timeout': -1, 'use_critic': True, 'use_gae': True, 'lambda': 1.0, 'kl_coeff': 0.2, 'sgd_minibatch_size': 128, 'shuffle_sequences': True, 'num_sgd_iter': 30, 'lr_schedule': None, 'vf_loss_coeff': 1.0, 'entropy_coeff': 0.0, 'entropy_coeff_schedule': None, 'clip_param': 0.3, 'vf_clip_param': 10.0, 'grad_clip': None, 'kl_target': 0.01, 'vf_share_layers': -1}