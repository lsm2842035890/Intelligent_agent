from gym import spaces, core
import os, glob
import random
import json
import numpy as np
import logging
import yaml
import math
import itertools

class D2RLTrainingEnv(core.Env):
    def __init__(self, yaml_conf):
        data_folders = [yaml_conf["root_folder"] + folder for folder in yaml_conf["data_folders"]]
        data_folder_weights = yaml_conf["data_folder_weights"]
        self.da_fo = ""
        self.yaml_conf = yaml_conf
        self.action_space = spaces.Box(low=0, high=1, shape=(24,))       # 周围六辆车左转右转加速减速对应的概率 6*4
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(42,))  # 周围六辆车的信息and 自车 7*6
        self.constant, self.weight_reward, self.exposure, self.positive_weight_reward = 0, 0, 0, 0  # some customized metric logging
        self.total_episode, self.total_steps = 0, 0
        if isinstance(data_folders, list):
            data_folder = random.choices(data_folders, weights=data_folder_weights)[0]
            self.da_fo = data_folder
        else:
            data_folder = data_folders
            self.da_fo = data_folder
        # 修改了
        self.crash_data_path_list= self.get_path_list(data_folder)
        self.all_data_path_list = self.crash_data_path_list
        self.episode_data_path = ""
        self.episode_data = None
        self.unwrapped.trials = 100
        self.unwrapped.reward_threshold = 1.5
        # 发生事故的两辆车的id
        self.coll_vehicles = []
        # 主责车的id
        self.not_main_responsibility_veh_id = ""
        self.cur_action = []

    def get_path_list(self, data_folder):
        crash_path = os.path.join(data_folder, "episode_only_crash")
        if os.path.isdir(crash_path):
            crash_data_path_list = glob.glob(crash_path + "/*.json")
        else:
            crash_data_path_list = []
        return crash_data_path_list    
    
    def reset(self, episode_data_path=None):
        self.constant, self.weight_reward, self.exposure, self.positive_weight_reward = 0, 0, 0, 0
        # self.total_episode = 0
        self.total_steps = 0
        self.episode_data_path = ""
        self.episode_data = None
        self.coll_vehicles = []
        self.not_main_responsibility_veh_id = ""
        self.cur_action = []
        return self._reset(episode_data_path)  
    
    def _reset(self, episode_data_path=None):
        self.total_episode += 1
        if not episode_data_path:
            self.episode_data_path = self.sample_data_this_episode()
        else:
            self.episode_data_path = episode_data_path
        # print("episode_data_path:", self.episode_data_path)    
        with open(self.episode_data_path) as data_file:
            self.episode_data = json.load(data_file)
        self.coll_vehicles = self.episode_data["collision_ids"]
        # episode_num = self.episode_data["episode_info"]["id"]
        # self.episode_data_path = os.path.join(self.da_fo, "episode", f"episode_{episode_num}.json")
        with open(self.episode_data_path) as data_file:
            self.episode_data = json.load(data_file)
        # print("episode_data_path:", self.episode_data_path)
        self.not_main_responsibility_veh_id = self.check_main_resp()
        # print("not_main_responsibility_veh_id:", self.not_main_responsibility_veh_id)
        if self.episode_data is not None:
            all_obs = self.episode_data[self.not_main_responsibility_veh_id]
            time_step_list = list(all_obs.keys())  # 0.0 0.1 0.2.....
            if len(time_step_list)!=0:
                init_obs = all_obs[time_step_list[0]]  # ego... lead... leftlead... leftfoll.....
                vehs_ids_list = list(init_obs.keys())  # ego lead lefglead leftfoll.....
                # print(vehs_ids_list)
                six_vehs_info = [[0 for _ in range(6)] for _ in range(7)]
                index_veh = 0
                for i in vehs_ids_list:
                    if init_obs[i] is not None:
                        # print(init_obs[i]["velocity"])
                        six_vehs_info[index_veh][0] = init_obs[i]["velocity"]
                        six_vehs_info[index_veh][1] = init_obs[i]["acceleration"]
                        six_vehs_info[index_veh][2] = init_obs[i]["heading"]
                        six_vehs_info[index_veh][3] = init_obs[i]["distance"]
                        six_vehs_info[index_veh][4] = math.sqrt(init_obs[i]["position"][0]**2 + init_obs[i]["position"][1]**2)
                        six_vehs_info[index_veh][5] = init_obs[i]["lane_index"]      
                    index_veh +=1    
                # return np.float32(six_vehs_info)
                return np.float32(list(itertools.chain(*six_vehs_info)))
            else:
                return self._reset()
        else:
            return self._reset()
        
    def sample_data_this_episode(self):
        episode_data_path = random.choices(self.crash_data_path_list, weights=[1] * len(self.crash_data_path_list))[0]
        # print("episode_data_path:", episode_data_path)
        return episode_data_path

    def step(self, action):
        # print("action:", action)
        # action = action.item()
        obs = self._get_observation()
        done, _ = self._get_done()
        self.cur_action.append(action)
        reward = self._get_reward()
        info = self._get_info()
        self.total_steps += 1
        return obs, reward, done, info  

    def _get_observation(self):
        all_obs = self.episode_data[self.not_main_responsibility_veh_id]
        time_step_list = list(all_obs.keys())  # 0.0 0.1 0.2.....
        current_obs = all_obs[time_step_list[self.total_steps]]  # ego lead leftlead leftfoll.....
        vehs_ids_list = list(current_obs.keys())  # ego lead lefglead leftfoll.....
        six_vehs_info = [[0 for _ in range(6)] for _ in range(7)]
        index_veh = 0
        for i in vehs_ids_list:
            if current_obs[i] is not None:
                six_vehs_info[index_veh][0] = current_obs[i]["velocity"]
                six_vehs_info[index_veh][1] = current_obs[i]["acceleration"]
                six_vehs_info[index_veh][2] = current_obs[i]["heading"]
                six_vehs_info[index_veh][3] = current_obs[i]["distance"]
                six_vehs_info[index_veh][4] = math.sqrt(current_obs[i]["position"][0]**2 + current_obs[i]["position"][1]**2)
                six_vehs_info[index_veh][5] = current_obs[i]["lane_index"] 
            index_veh +=1   
        # return np.float32(six_vehs_info) 
        return np.float32(list(itertools.chain(*six_vehs_info)))     
        # vehs_ids_list = vehs_ids_list[1:]

        # relative_speed_and_position = []
        # temp_length_two_vehs = 200
        # nearest_veh_id = ""

        # for i in range(0, 6):
        #     if current_obs[vehs_ids_list[i]]:
        #         # dis_temp = init_obs[vehs_ids_list[i]]["distance"]
        #         # six_dis.append(dis_temp)
        #         if current_obs[vehs_ids_list[i]]["distance"] < temp_length_two_vehs:
        #                 nearest_veh_id = vehs_ids_list[i]
        #                 temp_length_two_vehs = current_obs[vehs_ids_list[i]]["distance"]
        #         # six_dis.append(-9)  # 不存在那辆车的话距离-9 代表 无
        # # print(six_dis)
        # # return np.float32(six_dis)  # 返回周围六辆车的距离作为初始观察
        # if nearest_veh_id:
        #     relative_speed_and_position.append(abs(current_obs[nearest_veh_id]["velocity"] - current_obs["Ego"]["velocity"]))
        #     relative_speed_and_position.append(current_obs[nearest_veh_id]["distance"])
        # else:
        #     relative_speed_and_position = [200,200]    
        # return np.float32(relative_speed_and_position)

    def _get_done(self):
        stop = False
        reason = None
        if self.total_steps == len(self.episode_data[self.not_main_responsibility_veh_id].keys()) - 1:
            stop = True
            # if self.episode_data["collision_result"]:
            #     reason = {1: "CAV and BV collision"}
            # else:
            #     reason = {4: "CAV safely exist"}
        # return stop,reason
        return stop,reason    
    
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
        history_action = episode_data[self.not_main_responsibility_veh_id]
        time_step_list = list(history_action.keys())
        vehs_ids = ["Lead","LeftLead","RightLead","Foll","LeftFoll","RightFoll"]
        for i in range(1,len(agent_action)):
            model_output_matrix = agent_action[i]
            real_output_matrix = [[0 for _ in range(4)] for _ in range(6)]
            for j in range(0,6):
                if history_action[time_step_list[i]][vehs_ids[j]] is not None:
                    temp_id = history_action[time_step_list[i]][vehs_ids[j]]["veh_id"]
                    temp_info = episode_data[temp_id][time_step_list[i]]["Ego"]
                    if temp_info["prev_action"]["lateral"]=="left":
                        real_output_matrix[j][0] = 1
                        real_output_matrix[j][1] = 0
                        real_output_matrix[j][2] = 0
                        real_output_matrix[j][3] = 0
                    elif temp_info["prev_action"]["lateral"]=="right":
                        real_output_matrix[j][0] = 0
                        real_output_matrix[j][1] = 1
                        real_output_matrix[j][2] = 0
                        real_output_matrix[j][3] = 0
                    elif temp_info["prev_action"]["lateral"]=="central" and temp_info["prev_action"]["longitudinal"]>0:
                        real_output_matrix[j][0] = 0
                        real_output_matrix[j][1] = 0
                        real_output_matrix[j][2] = 1
                        real_output_matrix[j][3] = 0
                    else:
                        real_output_matrix[j][0] = 0
                        real_output_matrix[j][1] = 0
                        real_output_matrix[j][2] = 0
                        real_output_matrix[j][3] = 1
            # print("real_output_matrix",real_output_matrix,"model_output_matrix",model_output_matrix)            
            similarity = np.sqrt(np.sum((np.array(model_output_matrix) - np.array(real_output_matrix).flatten().tolist())**2)) 
            reward += 1-(similarity/24)     
        # print("reward",reward) 
        return reward  
 
    def _get_info(self):
        return {}

    def close(self):
        return

    def render(self):
        return
      
    def check_main_resp(self):       # 得到主责车辆 并且返回不是主责车辆的id
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
                    # return id_1
                    main_resp_id = id_1
                elif id_2_heading != 90.0 and id_2_pos > id_1_pos:
                    # print('233333')
                    # return id_2
                    main_resp_id = id_2
            else:
                if id_1_pos > id_2_pos:
                    # print('3')
                    # return id_2
                    main_resp_id = id_2
                else:
                    # print('4')
                    # return id_1
                    main_resp_id = id_1
        if main_resp_id != "":
            index = self.coll_vehicles.index(main_resp_id)
            index = 1-index
            return self.coll_vehicles[index]   
        else:
            return random.choices([id_1,id_2], weights=[1,1])[0]
        # print('5')
        # return random.choices([id_1,id_2], weights=[1,1])[0]

if __name__ == "__main__":
    with open(
            r"C:\Users\28420\Desktop\SAIC\Dense-Deep-Reinforcement-Learning-main\sumo_data_merge_400_highway_global_info_road_struc_agent_train\train_merge_400_s_d_global_info_road_struc.yaml",
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
