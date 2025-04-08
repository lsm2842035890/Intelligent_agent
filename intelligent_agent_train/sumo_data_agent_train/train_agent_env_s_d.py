from gym import spaces, core
import os, glob
import random
import json
import numpy as np
import logging
import yaml


class D2RLTrainingEnv(core.Env):
    def __init__(self, yaml_conf):
        data_folders = [yaml_conf["root_folder"] + folder for folder in yaml_conf["data_folders"]]
        data_folder_weights = yaml_conf["data_folder_weights"]
        self.da_fo = ""
        self.yaml_conf = yaml_conf
        self.action_space = spaces.Box(low=0.001, high=0.999, shape=(1,))
        self.observation_space = spaces.Box(low=-10, high=200, shape=(6,))  # 是6个周围车对自己的距离
        self.constant, self.weight_reward, self.exposure, self.positive_weight_reward = 0, 0, 0, 0  # some customized metric logging
        self.total_episode, self.total_steps = 0, 0
        if isinstance(data_folders, list):
            data_folder = random.choices(data_folders, weights=data_folder_weights)[0]
            self.da_fo = data_folder
        else:
            data_folder = data_folders
            self.da_fo = data_folder
        # 修改了
        self.crash_data_path_list, self.safe_data_path_list = self.get_path_list(data_folder)
        self.all_data_path_list = self.crash_data_path_list + self.safe_data_path_list
        self.episode_data_path = ""
        self.episode_data = None
        self.unwrapped.trials = 100
        self.unwrapped.reward_threshold = 1.5
        # 发生事故的两辆车的id
        self.coll_vehicles = []
        # 主责车的id
        self.main_responsibility_veh_id = ""
        self.cur_action = []

    def get_path_list(self, data_folder):
        # crash_target_weight_list = None
        # print(os.path.exists(data_folder + "/crash_weight_dict.json"))
        # if os.path.exists(data_folder + "/crash_weight_dict.json"):
        #     with open(data_folder + "/crash_weight_dict.json") as data_file:
        #         crash_weight_dict = json.load(data_file)
        #         self.crash_weight_dict = crash_weight_dict
        #         crash_data_path_list = list(crash_weight_dict.keys())
        #         crash_data_weight_list = [crash_weight_dict[path][0] for path in crash_data_path_list]
        # else:
        #     raise ValueError("No weight information!")
        crash_path = os.path.join(data_folder, "crash")
        # print(crash_path)
        if os.path.isdir(crash_path):
            crash_data_path_list = glob.glob(crash_path + "/*.json")
        else:
            crash_data_path_list = []

        tested_but_safe_path = os.path.join(data_folder, "tested_and_safe")
        if os.path.isdir(tested_but_safe_path):
            safe_data_path_list = glob.glob(tested_but_safe_path + "/*.json")
        else:
            safe_data_path_list = []
        logging.info(f'{len(crash_data_path_list)} Crash Events, {len(safe_data_path_list)} Safe Events')
        return crash_data_path_list, safe_data_path_list


    def reset(self, episode_data_path=None):
        self.constant, self.weight_reward, self.exposure, self.positive_weight_reward = 0, 0, 0, 0
        # self.total_episode = 0
        self.total_steps = 0
        self.episode_data_path = ""
        self.episode_data = None
        self.coll_vehicles = []
        self.main_responsibility_veh_id = ""
        self.cur_action = []
        return self._reset(episode_data_path)


    def _reset(self, episode_data_path=None):
        self.total_episode += 1
        if not episode_data_path:
            self.episode_data_path = self.sample_data_this_episode()
        else:
            self.episode_data_path = episode_data_path
        with open(self.episode_data_path) as data_file:
            self.episode_data = json.load(data_file)
        self.coll_vehicles = self.episode_data["collision_id"]
        episode_num = self.episode_data["episode_info"]["id"]
        self.episode_data_path = os.path.join(self.da_fo, "episode", f"episode_{episode_num}.json")
        with open(self.episode_data_path) as data_file:
            self.episode_data = json.load(data_file)
        self.main_responsibility_veh_id = self.check_main_resp()
        if self.episode_data is not None:
            all_obs = self.episode_data[self.main_responsibility_veh_id]
            time_step_list = list(all_obs.keys())  # 0.0 0.1 0.2.....
            if len(time_step_list):
                init_obs = all_obs[time_step_list[0]]  # ego lead leftlead leftfoll.....
                vehs_ids_list = list(init_obs.keys())  # ego lead lefglead leftfoll.....
                vehs_ids_list = vehs_ids_list[1:]
                six_dis = []
                for i in range(0, 6):
                    if init_obs[vehs_ids_list[i]]:
                        dis_temp = init_obs[vehs_ids_list[i]]["distance"]
                        six_dis.append(dis_temp)
                    else:
                        six_dis.append(-9)  # 不存在那辆车的话距离-9 代表 无
                # print(six_dis)
                return np.float32(six_dis)  # 返回周围六辆车的距离作为初始观察
            else:
                return self._reset()
        else:
            return self._reset()
        # with open(self.episode_data_path) as data_file:
        #     self.episode_data = self.filter_episode_data(json.load(data_file))
        # 等待修改
        # 检查谁是主责=肇事车辆 self.coll_vehicles = [id1,id2] self.main_responsibility_veh_id = id1 def check_main_resp(crash.json,all_vehicle_step_info.josn)
        # 得到肇事车辆的初始obs
        #
        # if self.episode_data is not None:E:\pycharmcode\sumo_data_generation\Experiment-sumo_data_gen_2024-06-22\crash
        #     all_obs = self.episode_data["drl_obs_step_info"]
        #     time_step_list = list(all_obs.keys())
        #     if len(time_step_list):
        #         init_obs = np.float32(all_obs[time_step_list[0]])
        #         return init_obs
        #     else:
        #         return self._reset()
        # else:
        #     return self._reset()

    def sample_data_this_episode(self):
        episode_data_path = random.choices(self.crash_data_path_list, weights=[1] * len(self.crash_data_path_list))[0]
        return episode_data_path


    def step(self, action):
        action = action.item()
        obs = self._get_observation()
        done, _ = self._get_done()
        time_step_list = list(self.episode_data[self.main_responsibility_veh_id].keys())
        # criticality_this_step = self.episode_data["criticality_step_info"][time_step_list[self.total_steps]]
        # self.episode_data["drl_epsilon_step_info"][time_step_list[self.total_steps]] = action
        self.cur_action.append(action)
        reward = self._get_reward()
        info = self._get_info()
        self.total_steps += 1
        return obs, reward, done, info


    def _get_info(self):
        return {}


    def close(self):
        return


    def _get_observation(self):
        all_obs = self.episode_data[self.main_responsibility_veh_id]
        time_step_list = list(all_obs.keys())  # 0.0 0.1 0.2.....
        current_obs = all_obs[time_step_list[self.total_steps]]  # ego lead leftlead leftfoll.....
        vehs_ids_list = list(current_obs.keys())  # ego lead lefglead leftfoll.....
        vehs_ids_list = vehs_ids_list[1:]
        six_dis = []
        for i in range(0, 6):
            if current_obs[vehs_ids_list[i]]:
                dis_temp = current_obs[vehs_ids_list[i]]["distance"]
                six_dis.append(dis_temp)
            else:
                six_dis.append(-9)
        return np.float32(six_dis)
        # all_obs = self.episode_data["drl_obs_step_info"]
        # time_step_list = list(all_obs.keys())
        # try:
        #     obs = np.float32(all_obs[time_step_list[self.total_steps]])
        # except:
        #     print(self.total_steps, time_step_list)
        #     obs = np.float32(all_obs[time_step_list[-1]])
        # return obs


    def _get_reward(self):
        reward = 0
        stop, reason = self._get_done()
        if not stop:
            return 0
        else:
            reward = self.getreward(self.episode_data,self.cur_action)
            # drl_epsilon_weight = self._get_drl_epsilon_weight(self.episode_data["weight_step_info"],
            #                                                   self.episode_data["drl_epsilon_step_info"],
            #                                                   self.episode_data["ndd_step_info"],
            #                                                   self.episode_data["criticality_step_info"])
        #     if 1 in reason:
        #         print(self.episode_data["drl_epsilon_step_info"])
        #         adv_action_num = self.get_multiple_adv_action_num(self.episode_data["weight_step_info"])
        #         if adv_action_num > 1:
        #             return 0  # if multiple adversarial action is detected, this episode will be of no use
        #         clip_reward_threshold = self.yaml_conf["clip_reward_threshold"]
        #         q_amplifier_reward = clip_reward_threshold - drl_epsilon_weight * 500 * clip_reward_threshold  # drl epsilon weight reward
        #         if q_amplifier_reward < -clip_reward_threshold:
        #             q_amplifier_reward = -clip_reward_threshold
        #         print("final_reward:", q_amplifier_reward)
        #         return q_amplifier_reward
        #     else:
        #         return 0
        return reward

    def getreward(self,episode_data,agent_action):   #用主责车辆历史加速度数据和模型得到的
        reward = 0
        history_action = episode_data[self.main_responsibility_veh_id]
        time_step_list = list(history_action.keys())
        # print(time_step_list)
        # print(len(time_step_list),len(agent_action))
        for i in range(0,len(agent_action)):
            # print(history_action[time_step_list[i]]["Ego"]["prev_action"])
            if history_action[time_step_list[i]]["Ego"]["prev_action"]:
                one_step_history_action = history_action[time_step_list[i]]["Ego"]["prev_action"]["longitudinal"]
                if one_step_history_action:
                    reward += abs(((one_step_history_action + 4) / 6 - agent_action[i]))
                else:
                    reward += abs(((0 + 4) / 6 - agent_action[i]))
            else:
                reward = abs(((0 + 4) / 6 - agent_action[i]))

            # one_step_history_action = history_action[time_step_list[i]]["Ego"]["prev_action"]["longitudinal"]
            # if not one_step_history_action :
            #     one_step_history_action = 0
            # reward += abs(((one_step_history_action+4)/6 - agent_action[i]))
        return reward

    def _get_done(self):
        stop = False
        reason = None
        if self.total_steps == len(self.episode_data[self.main_responsibility_veh_id].keys()) - 1:
            stop = True
            # if self.episode_data["collision_result"]:
            #     reason = {1: "CAV and BV collision"}
            # else:
            #     reason = {4: "CAV safely exist"}
        # return stop,reason
        return stop,reason


    def render(self):
        return

    def check_main_resp(self):
        main_resp_id = ""
        id_1 = self.coll_vehicles[0]
        id_2 = self.coll_vehicles[1]
        id_1_data = self.episode_data[id_1]
        id_1_timelist = list(id_1_data.keys())
        # print('id_1_timelist',id_1_timelist)
        id_2_data = self.episode_data[id_2]
        id_2_timelist = list(id_2_data.keys())
        # print('id_2_timelist', id_2_timelist)
        min_timelist_num = len(id_1_timelist) if len(id_1_timelist)<len(id_2_timelist) else len(id_2_timelist)
        # print('min_timelist_num',min_timelist_num)
        for i in range(0,min_timelist_num-1):
            id_1_heading = id_1_data[id_1_timelist[min_timelist_num-1-i]]["Ego"]["heading"]
            id_2_heading = id_2_data[id_2_timelist[min_timelist_num-1-i]]["Ego"]["heading"]
            # print('id_1_heading',id_1_heading,'id_2_heading',id_2_heading)
            id_1_pos = id_1_data[id_1_timelist[min_timelist_num-1-i]]["Ego"]["position"][0]
            id_2_pos = id_2_data[id_2_timelist[min_timelist_num-1-i]]["Ego"]["position"][0]
            # print('id_1_pos',id_1_pos,'id_2_pos',id_2_pos)
            if id_1_heading - id_2_heading != 0 :
                if id_1_heading != 90.0 and id_1_pos > id_2_pos:
                    # print('122222')
                    return id_1
                elif id_2_heading != 90.0 and id_2_pos > id_1_pos:
                    # print('233333')
                    return id_2
            else:
                if id_1_pos > id_2_pos:
                    # print('3')
                    return id_2
                else:
                    # print('4')
                    return id_1
        # print('5')
        return random.choices([id_1,id_2], weights=[1,1])[0]

if __name__ == "__main__":
    with open(
            r'C:\Users\28420\Desktop\SAIC\Dense-Deep-Reinforcement-Learning-main\sumo_data_agent_train\train_s_d.yaml',
            'r') as f:
        yaml_conf = yaml.load(f, Loader=yaml.FullLoader)
    env = D2RLTrainingEnv(yaml_conf)
    # env.reset()
    # print(env.crash_data_path_list)
    # print('_____________________')
    # print(env.safe_data_path_list)
    for i in range(100):
        obs = env.reset()
        # print(env.total_episode)
        # print('da_fo', env.da_fo)
        # print('coll_vehs', env.coll_vehicles)
        # print('episode_data_path', env.episode_data_path)
        while True:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            # print(reward)
            # print(env.total_steps)
            if done:
                break
        # print(reward)
        # print(len(env.cur_action),env.cur_action)
        # print('coll_vehs', env.coll_vehicles)
        # print('episode_data_path', env.episode_data_path)
        # print(env.total_episode)
