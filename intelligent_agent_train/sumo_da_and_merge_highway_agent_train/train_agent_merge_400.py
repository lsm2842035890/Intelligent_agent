import ray
from ray import tune
import glob
from ray.tune.registry import register_env
from train_agent_merge_400_env_s_d import D2RLTrainingEnv
import yaml, argparse
from tqdm import tqdm
import json

parser = argparse.ArgumentParser()
parser.add_argument('--yaml_conf', type=str, default=r'C:\Users\28420\Desktop\SAIC\Dense-Deep-Reinforcement-Learning-main\sumo_da_and_merge_highway_agent_train\train_merge_400_s_d.yaml', metavar='N',
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
        "num_gpus": 0,
        "num_workers": yaml_conf["num_workers"],
        "num_envs_per_worker": 1,
        "gamma": 1.0,
        "rollout_fragment_length": 600,
        "vf_clip_param": yaml_conf["clip_reward_threshold"],
        "framework": "torch",
        "ignore_worker_failures": True,
        "callbacks": MyCallbacks,
        # #######
        # "model": {
        #     # 启用LSTM
        #     "use_lstm": True,
        #     # LSTM序列最大长度，默认20
        #     "max_seq_len": 20,
        #     # LSTM单元的大小，默认256
        #     "lstm_cell_size": 256,
        #     # 是否使用前一个动作作为LSTM的输入
        #     "lstm_use_prev_action": False,
        #     # 是否使用前一个奖励作为LSTM的输入
        #     "lstm_use_prev_reward": False,
        #     # LSTM数据格式：时间优先 (TxBx..) 或批次优先 (BxTx..)
        #     "_time_major": False,
        # },
        # #####
    },
    checkpoint_freq=100,
    local_dir=yaml_conf["local_dir"],
    name=yaml_conf["experiment_name"],
)