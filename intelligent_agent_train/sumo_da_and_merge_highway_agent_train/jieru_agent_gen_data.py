import json
import os, sys
import shutil
libsumo_flag = (not True)
if libsumo_flag:
    import libsumo as traci
else:
    import traci
# import traci
import traci.constants as tc
import sumolib
import numpy as np
import math
from abc import ABC
import random
import torch
import warnings

class torch_discriminator_agent:
    def __init__(self, checkpoint_path):
        if not checkpoint_path:
            # checkpoint_path = "./model.pt"
            checkpoint_path = r"E:\pycharmcode\sumo_data_generation\checkpoints_400_merge_highway_agent\model.pt"
        # print("Loading checkpoint", checkpoint_path)
        self.model = torch.jit.load(checkpoint_path)
        self.model.eval()

    def compute_action(self, observation):
        warnings.filterwarnings("ignore", category=UserWarning)
        # 忽略特定类型的警告
        lb = 0.001
        ub = 0.999
        obs = torch.reshape(torch.tensor(observation,dtype=torch.float32), (1,len(observation)))
        # obs = torch.reshape(torch.tensor(agent_input,dtype=torch.float32), (1,len(agent_input)))
        out = self.model({"obs":obs},[torch.tensor([0.0])],torch.tensor([1]))
        # out = self.model({"obs":obs})
        # if simulation_config["epsilon_type"] == "discrete":
        #     action = torch.argmax(out[0][0])
        # else:
        #     action = np.clip((float(out[0][0][0])+1)*(ub-lb)/2 + lb, lb, ub)
        return out

class Observation(ABC):
    """Observation class store the vehicle observations, the time_stamp object is essential to allow observation to only update once. 
    It is composed of the local information, context information, processed information and time stamp.
    local: a dictionary{ vehicle ID: subsribed results (dictionary)
    }
    context: a dictionary{ vehicle ID: subsribed results (dictionary)
    }
    information: a dictionary{
        'Ego': {'veh_id': vehicle ID, 'speed': vehicle velocity [m/s], 'position': tuple of X,Y coordinates [m], 'heading': vehicle angle [degree], 'lane_index': lane index of vehicle, 'distance': 0 [m]},
        'Lead': {'veh_id': vehicle ID, 'speed': vehicle velocity [m/s], 'position': tuple of X,Y coordinates [m], 'heading': vehicle angle [degree], 'lane_index': lane index of vehicle, 'distance': range between vehicles [m]} or None
        'Foll': {'veh_id': vehicle ID, 'speed': vehicle velocity [m/s], 'position': tuple of X,Y coordinates [m], 'heading': vehicle angle [degree], 'lane_index': lane index of vehicle, 'distance': range between vehicles [m]} or None
        'LeftLead': {'veh_id': vehicle ID, 'speed': vehicle velocity [m/s], 'position': tuple of X,Y coordinates [m], 'heading': vehicle angle [degree], 'lane_index': lane index of vehicle, 'distance': range between vehicles [m]} or None
        'RightLead': {'veh_id': vehicle ID, 'speed': vehicle velocity [m/s], 'position': tuple of X,Y coordinates [m], 'heading': vehicle angle [degree], 'lane_index': lane index of vehicle, 'distance': range between vehicles [m]} or None
        'LeftFoll': {'veh_id': vehicle ID, 'speed': vehicle velocity [m/s], 'position': tuple of X,Y coordinates [m], 'heading': vehicle angle [degree], 'lane_index': lane index of vehicle, 'distance': range between vehicles [m]} or None
        'RightFoll': {'veh_id': vehicle ID, 'speed': vehicle velocity [m/s], 'position': tuple of X,Y coordinates [m], 'heading': vehicle angle [degree], 'lane_index': lane index of vehicle, 'distance': range between vehicles [m]} or None
    }
    time_stamp is used to record simulation time and for lazy use.
    """
    def __init__(self, ego_id=None, time_stamp=-1):
        if not ego_id:
            raise ValueError("No ego vehicle ID is provided!")
        self.ego_id = ego_id
        self.information = None
        if time_stamp == -1:
            raise ValueError("Observation is used before simulation started!")
        self.time_stamp = time_stamp

    def update(self,subscription =None,prev_action=None):     #get_vehicle_context_subscription_results
    # """Update the observation of vehicle.

    # Args:
    #     subscription (dict, optional): Context information obtained from SUMO Traci. Defaults to None.

    # Raises:
    #     ValueError: When supscription results are None, raise error.
    # """ 
        if not subscription:
            raise ValueError("No subscription results are provided!")       
        self.information = self.traci_based_process(subscription,prev_action)

    def traci_based_process(self,subscription = None,prev_action=None):
        obs = {"Ego": Observation.pre_process_subscription(subscription, self.ego_id, prev_action)}
        obs["Lead"] = get_leading_vehicle(self.ego_id)
        obs["LeftLead"] = get_neighboring_leading_vehicle(self.ego_id, "left")
        obs["RightLead"] = get_neighboring_leading_vehicle(self.ego_id, "right")
        obs["Foll"] = get_following_vehicle(self.ego_id)
        obs["LeftFoll"] = get_neighboring_following_vehicle(self.ego_id, "left")
        obs["RightFoll"] = get_neighboring_following_vehicle(self.ego_id, "right")
        return obs
    
    @staticmethod
    # @profile
    def pre_process_subscription(subscription, veh_id=None, prev_action=None,distance=0.0):
        """Modify the subscription results into a standard form.

        Args:
            subscription (dict): Context subscription results of vehicle.
            simulator (Simulator): Simulator object.
            veh_id (str, optional): Vehicle ID. Defaults to None.
            distance (float, optional): Distance from the ego vehicle [m]. Defaults to 0.0.

        Returns:
            dict: Standard for of vehicle information.
        """
        if not veh_id:
            return None
        veh = {"veh_id": veh_id}

        veh["could_drive_adjacent_lane_left"] = get_vehicle_lane_adjacent(veh_id,1)
        veh["could_drive_adjacent_lane_right"] = get_vehicle_lane_adjacent(veh_id,-1)
        veh["distance"] = distance
        veh["heading"] = subscription[veh_id][67]
        veh["lane_index"] = subscription[veh_id][82]
        veh["lateral_speed"] = subscription[veh_id][50]
        veh["lateral_offset"] = subscription[veh_id][184]
        # veh["prev_action"] = vehicle.controller.action   #prev_action
        veh["prev_action"] = prev_action
        veh["position"] = subscription[veh_id][66]
        veh["position3D"] = subscription[veh_id][57]
        veh["velocity"] = subscription[veh_id][64]
        veh["road_id"] = subscription[veh_id][80]
        veh["acceleration"] = subscription[veh_id][114]
        return veh    
    
def get_vehicle_context_subscription_results(vehID):     #subscription method
    """Get subscription results of the context information.

    Args:
        vehID (str): Vehicle ID.

    Returns:
        dict: Context subscription results.
    """        
    return traci.vehicle.getContextSubscriptionResults(vehID)

def _get_observation(ego_id, time_stamp,prev_action):
    obs = Observation(ego_id, time_stamp)
    subscribtion = get_vehicle_context_subscription_results(ego_id)
    obs.update(subscription=subscribtion,prev_action=prev_action)
    return obs.information

def act(id, action, veh_info, time_step, all_timestep_list, lc_duration=1, step_size=0.1, action_step_size=0.1):
    """Vehicle acts based on the input action.

    Args:
        action (dict): Lonitudinal and lateral actions. It should have the format: {'longitudinal': float, 'lateral': str}. The longitudinal action is the longitudinal acceleration, which should be a float. The lateral action should be the lane change direction. 'central' represents no lane change. 'left' represents left lane change, and 'right' represents right lane change.
    """
    traci.vehicle.setSpeedMode(id, 0)
    traci.vehicle.setLaneChangeMode(id, 0)
    controlled_acc = action["longitudinal"]
    # current_velocity = veh_info[all_timestep_list[time_step - 1]]["Ego"]["velocity"]
    current_velocity = traci.vehicle.getSpeed(id)
    if current_velocity + controlled_acc > 40:
        controlled_acc = 40 - current_velocity
    elif current_velocity + controlled_acc < 20:
        controlled_acc = 20 - current_velocity

    if action["lateral"] == "central":
        current_lane_offset = get_vehicle_lateral_lane_position(id)
        change_vehicle_sublane_dist(id, -current_lane_offset, 0.1)
        change_vehicle_speed(id, controlled_acc, 1, 0.1)
    else:
        change_vehicle_lane(id, action["lateral"], duration=1)
        change_vehicle_speed(id, controlled_acc, duration=1)

def get_vehicle_lateral_lane_position(vehID):
    return traci.vehicle.getLateralLanePosition(vehID)

def change_vehicle_sublane_dist(vehID, latdist, duration):
    lat_max_v = _cal_lateral_maxSpeed(vehID, abs(latdist), duration)
    traci.vehicle.setMaxSpeedLat(vehID, lat_max_v)
    traci.vehicle.changeSublane(vehID, latdist)

def change_vehicle_speed(vehID, acceleration, duration=1.0, step_size=0.1):
    init_speed = traci.vehicle.getSpeed(vehID)
    final_speed = init_speed + acceleration * (step_size + duration)
    if final_speed < 0:
        final_speed = 0
    traci.vehicle.slowDown(vehID, final_speed, duration)

def change_vehicle_lane(vehID, direction, duration=1.0, sublane_flag=True, step_size=0.1):
    if sublane_flag:
        latdist = _cal_lateral_distance(vehID, direction)
        lat_max_v = _cal_lateral_maxSpeed(vehID, abs(latdist), duration)
        # print("lat_max_v:", lat_max_v, "latdist:", latdist)
        traci.vehicle.setMaxSpeedLat(vehID, lat_max_v)
        traci.vehicle.changeSublane(vehID, latdist)
    else:
        if direction == "left":
            indexOffset = 1
        elif direction == "right":
            indexOffset = -1
        else:
            raise ValueError("Unknown direction for lane change command")
            # print("Unknown direction for lane change command")
        traci.vehicle.changeLaneRelative(vehID, indexOffset, step_size)     

def get_available_lanes(sumo_net_file_path, edge_id=None):
    """Get the available lanes in the sumo network

    Returns:
        list(sumo lane object): Possible lanes to insert vehicles
    """
    sumo_net = sumolib.net.readNet(sumo_net_file_path)
    if edge_id == None:
        sumo_edges = sumo_net.getEdges()
    else:
        sumo_edges = [sumo_net.getEdge(edge_id)]
    available_lanes = []
    for edge in sumo_edges:
        for lane in edge.getLanes():
            available_lanes.append(lane)
    return available_lanes

def _cal_lateral_maxSpeed(vehID, lane_width, time=1.0):
    """Calculate the maximum lateral speed for lane change maneuver.

    Args:
        vehID (str): Vehicle ID.
        lane_width (float): Width of the lane.
        time (float, optional): Specified time interval to complete the lane change maneuver in s. Defaults to 1.0.

    Raises:
        ValueError: If the maximum lateral acceleration of the vehicle is too small, it is impossible to complete the lane change maneuver in the specified duration.

    Returns:
        float: Maximum lateral speed aiming to complete the lane change behavior in the specified time duration.
    """
    # accelerate laterally to the maximum lateral speed and maintain
    # v^2 - b*v + c = 0
    lat_acc = float(traci.vehicle.getParameter(vehID, 'laneChangeModel.lcAccelLat'))
    b, c = lat_acc * (time), lat_acc * lane_width
    delta_power = b ** 2 - 4 * c
    if delta_power >= 0:
        lat_max_v = (-math.sqrt(delta_power) + b) / 2
    else:
        raise ValueError("The lateral maximum acceleration is too small.")
    return lat_max_v

def _cal_lateral_distance(vehID, direction):
    """Calculate lateral distance to the target lane for a complete lane change maneuver.

    Args:
        vehID (str): Vehicle ID.
        direction (str): Direction, i.e. "left" and "right".

    Raises:
        ValueError: Unknown lane id.
        ValueError: Unknown lane id.
        ValueError: Unknown direction.

    Returns:
        float: Distance in m.
    """
    origin_lane_id = traci.vehicle.getLaneID(vehID)
    edge_id = traci.vehicle.getRoadID(vehID)
    lane_index = int(origin_lane_id.split('_')[-1])
    origin_lane_width = traci.lane.getWidth(origin_lane_id)
    if edge_id == "E0" or edge_id == "E1":
        if direction == "left":
            if lane_index == 0:
                lane_index = 1
            else:
                return 0     
            target_lane_id = edge_id + "_" + str(lane_index)
            try:
                target_lane_width = traci.lane.getWidth(target_lane_id)
            except:
                raise ValueError("Unknown lane id: " + target_lane_id + " in the lane change maneuver.")    
            latdist = (origin_lane_width + target_lane_width) / 2
        elif direction == "right":
            if lane_index == 1:
                lane_index = 0
            else:
                return 0
            target_lane_id = edge_id + "_" + str(lane_index)
            try:
                target_lane_width = traci.lane.getWidth(target_lane_id)
            except:
                raise ValueError("Unknown lane id: " + target_lane_id + " in the lane change maneuver.")
            latdist = -(origin_lane_width + target_lane_width) / 2
        else:
            raise ValueError("Unknown direction for lane change command")
        return latdist
    elif edge_id == "E2":
        # target_lane_id = edge_id + "_" + str(lane_index)
        # target_lane_width = traci.lane.getWidth(target_lane_id)
        # latdist = -(origin_lane_width + target_lane_width) / 2
        return 0
    else:
        return 0
        raise ValueError("Unknown edge id: " + edge_id + " in the lane change maneuver.")

    # if direction == "left":
    #     if lane_index == 1:
    #         lane_index = 0
    #     target_lane_id = edge_id + "_" + str(lane_index + 1)
    #     try:
    #         target_lane_width = traci.lane.getWidth(target_lane_id)
    #     except:
    #         raise ValueError("Unknown lane id: " + target_lane_id + " in the lane change maneuver.")   
    #         # target_lane_width = 4
    #         # print("Unknown lane id: " + target_lane_id + " in the lane change maneuver.")    
    #     latdist = (origin_lane_width + target_lane_width) / 2
    # elif direction == "right":
    #     if lane_index == 0:
    #         lane_index = 1
    #     target_lane_id = edge_id + "_" + str(lane_index - 1)
    #     try:
    #         target_lane_width = traci.lane.getWidth(target_lane_id)
    #     except:
    #         raise ValueError("Unknown lane id: " + target_lane_id + " in the lane change maneuver.")
    #         # target_lane_width = 4
    #         # print("Unknown lane id: " + target_lane_id + " in the lane change maneuver.")
    #     latdist = -(origin_lane_width + target_lane_width) / 2
    # else:
    #     raise ValueError("Unknown direction for lane change command")
    #     # target_lane_width = 4
    #     # print("Unknown direction for lane change command")
    # return latdist    

def get_leading_vehicle(vehID):
    """Get the information of the leading vehicle.

    Args:
        vehID (str): ID of the ego vehicle.

    Returns:
        dict: necessary information of the leading vehicle, including:
            str: ID of the leading vehicle(accessed by 'veh_id'), 
            float: Leading vehicle speed (accessed by 'velocity'),
            tuple(float, float): Leading vehicle position in X and Y (accessed by 'position'),
            int: Leading vehicle lane index (accessed by 'lane_index')
            float: Distance between the ego vehicle and the leading vehicle (accessed by 'distance').
    """
    # get leading vehicle information: a list:
    # first element: leader id
    # second element: distance from leading vehicle to ego vehicle 
    # (it does not include the minGap of the ego vehicle)
    leader_info = traci.vehicle.getLeader(vehID, dist = 115) # empty leader: None
    if leader_info is None:
        return None
    else:
        r = leader_info[1] + traci.vehicle.getMinGap(vehID)
        return get_ego_vehicle(leader_info[0],r)
    
def get_ego_vehicle(vehID, dist = 0.0):
    """Get the information of the ego vehicle.

    Args:
        vehID (str): ID of the ego vehicle.
        dist (float, optional): Distance between two vehicles. Defaults to 0.0.

    Returns:
        dict: Necessary information of the ego vehicle, including:
            str: Vehicle ID (accessed by 'veh_id'), 
            float: Vehicle speed (accessed by 'velocity'),
            tuple(float, float): Vehicle position in X and Y (accessed by 'position'),
            int: Vehicle lane index (accessed by 'lane_index')
            float: Distance between the ego vehicle and another vehicle (accessed by 'distance').
    """        
    ego_veh = None
    if dist <= 115:
        ego_veh = {'veh_id':vehID}
        ego_veh['distance'] = dist
        try:
            # get ego vehicle information: a dict:
            # 66: position (a tuple); 64: velocity, 67: angle, 82: lane_index
            ego_info = traci.vehicle.getSubscriptionResults(vehID)
            ego_veh['velocity'] = ego_info[64]
            ego_veh['position'] = ego_info[66]
            ego_veh['heading'] = ego_info[67]
            ego_veh['lane_index'] = ego_info[82]
            ego_veh['position3D'] = ego_info[57]
            ego_veh["acceleration"] = ego_info[114]
        except:
            ego_veh['velocity'] = traci.vehicle.getSpeed(vehID)
            ego_veh['position'] = traci.vehicle.getPosition(vehID)
            ego_veh['heading'] = traci.vehicle.getAngle(vehID)
            ego_veh['lane_index'] = traci.vehicle.getLaneIndex(vehID)
            ego_veh['position3D'] = traci.vehicle.getPosition3D(vehID)
            ego_veh["acceleration"] = traci.vehicle.getAcceleration(vehID)
    return ego_veh    

def get_neighboring_leading_vehicle(vehID, dir):
    """Get the information of the neighboring leading vehicle.

    Args:
        vehID (str): ID of the ego vehicle.
        dir (str): Choose from "left" and "right".

    Returns:
        dict: necessary information of the neighboring leading vehicle, including:
            str: ID of the neighboring leading vehicle(accessed by 'veh_id'), 
            float: Neighboring leading vehicle speed (accessed by 'velocity'),
            tuple(float, float): Neighboring leading vehicle position in X and Y (accessed by 'position'),
            int: Neighboring leading vehicle lane index (accessed by 'lane_index')
            float: Distance between the ego vehicle and the neighboring leading vehicle (accessed by 'distance').
    """
    # get neighboring leading vehicle information: a list of tuple:
    # first element: leader id
    # second element: distance from leading vehicle to ego vehicle 
    # (it does not include the minGap of the ego vehicle)
    if dir == "left":
        leader_info = traci.vehicle.getNeighbors(vehID,2) # empty leftleader: len=0
    elif dir == "right":
        leader_info = traci.vehicle.getNeighbors(vehID,3) # empty rightleader: len=0
    else: 
        raise ValueError('NotKnownDirection')
    if len(leader_info) == 0:
        return None
    else:
        leader_info_list = [list(item) for item in leader_info]
        for i in range(len(leader_info)):
            leader_info_list[i][1] += traci.vehicle.getMinGap(vehID)
        sorted_leader = sorted(leader_info_list,key=lambda l:l[1])
        closest_leader = sorted_leader[0]
        return get_ego_vehicle(closest_leader[0],closest_leader[1])    

def get_following_vehicle(vehID):
    """Get the information of the following vehicle.

    Args:
        vehID (str): ID of the ego vehicle.

    Returns:
        dict: necessary information of the following vehicle, including:
            str: ID of the following vehicle(accessed by 'veh_id'), 
            float: Following vehicle speed (accessed by 'velocity'),
            tuple(float, float): Following vehicle position in X and Y (accessed by 'position'),
            int: Following vehicle lane index (accessed by 'lane_index')
            float: Distance between the ego vehicle and the following vehicle (accessed by 'distance').
    """
    # get following vehicle information: a list:
    # first element: follower id
    # second element: distance from ego vehicle to following vehicle 
    # (it does not include the minGap of the following vehicle)
    follower_info = traci.vehicle.getFollower(vehID, dist = 115) # empty follower: ('',-1) 
    if follower_info[1] == -1:
        return None
    else:
        r = follower_info[1] + traci.vehicle.getMinGap(follower_info[0])
        return get_ego_vehicle(follower_info[0],r)
    
def get_neighboring_following_vehicle(vehID, dir):
    """Get the information of the neighboring following vehicle.

    Args:
        vehID (str): ID of the ego vehicle.
        dir (str): Choose from "left" and "right".

    Returns:
        dict: necessary information of the neighboring following vehicle, including:
            str: ID of the neighboring following vehicle(accessed by 'veh_id'), 
            float: Neighboring following vehicle speed (accessed by 'velocity'),
            tuple(float, float): Neighboring following vehicle position in X and Y (accessed by 'position'),
            int: Neighboring following vehicle lane index (accessed by 'lane_index')
            float: Distance between the ego vehicle and the neighboring following vehicle (accessed by 'distance').
    """
    # get neighboring following vehicle information: a list of tuple:
    # first element: follower id
    # second element: distance from ego vehicle to following vehicle 
    # (it does not include the minGap of the following vehicle)
    if dir == "left":
        follower_info = traci.vehicle.getNeighbors(vehID,0) # empty leftfollower: len=0
    elif dir == "right":
        follower_info = traci.vehicle.getNeighbors(vehID,1) # empty rightfollower: len=0
    else: 
        raise ValueError('NotKnownDirection')
    if len(follower_info) == 0:
        return None
    else:
        follower_info_list = [list(item) for item in follower_info]
        for i in range(len(follower_info)):
            follower_info_list[i][1] += traci.vehicle.getMinGap(follower_info_list[i][0])
        sorted_follower = sorted(follower_info_list,key=lambda l:l[1])
        closest_follower = sorted_follower[0]
        return get_ego_vehicle(closest_follower[0],closest_follower[1])   
    
def get_vehicle_lane_adjacent(vehID, direction):
    """Get whether the vehicle is allowed to drive on the adjacent lane.

    Args:
        vehID (str): Vehicle ID.
        direction (int): 1 represents left, while -1 represents right.

    Returns:
        bool: Whether the vehicle can drive on the specific lane.
    """
    if direction not in [-1,1]:
        raise ValueError("Unknown direction input:"+str(direction))        
    lane_index = get_vehicle_lane_index(vehID)
    new_lane_index = lane_index+direction
    edge_id = get_vehicle_roadID(vehID)
    lane_num = get_edge_lane_number(edge_id)
    if new_lane_index < 0 or new_lane_index >=lane_num:
        # Adjacent lane does not exist.
        return False
    new_lane_id = edge_id+"_"+str(new_lane_index)
    veh_class = get_vehicle_class(vehID)
    disallowed = get_lane_disallowed(new_lane_id)
    return not veh_class in disallowed       

def get_vehicle_lane_index(vehID):
    """Get vehicle lane index.

    Args:
        vehID (str): Vehicle ID.

    Returns:
        int: Lane index.
    """        
    return traci.vehicle.getLaneIndex(vehID)       

def get_vehicle_roadID(vehID):
    """Get road ID where the vehicle is driving on.

    Args:
        vehID (str): Vehicle ID.

    Returns:
        str: Road ID.
    """        
    return traci.vehicle.getRoadID(vehID)

def get_edge_lane_number(edgeID):
    """Get lane number of the edge.

    Args:
        edgeID (str): Edge ID.

    Returns:
        int: Lane number.
    """        
    return traci.edge.getLaneNumber(edgeID)

def get_vehicle_class(vehID):
    """Get vehicle class.

    Args:
        vehID (str): Vehicle ID.

    Returns:
        str: Abstract vehicle class, such as "passenger".
    """        
    return traci.vehicle.getVehicleClass(vehID)

def get_lane_disallowed(laneID):
    """Get disallowed vehicle class of the lane.

    Args:
        laneID (str): Lane ID.

    Returns:
        list(str): Disallowed vehicle class, such as "passenger".
    """        
    return traci.lane.getDisallowed(laneID)

def subscribe_vehicle_all_information(vehID, max_obs_range=120):
    """Subscribe to store vehicle's complete information.

    Args:
        vehID (str): Vehicle ID.
    """
    subscribe_vehicle_ego(vehID)
    traci.vehicle.subscribeContext(vehID, tc.CMD_GET_VEHICLE_VARIABLE, max_obs_range, [tc.VAR_LENGTH, tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_LANE_INDEX, tc.VAR_ANGLE, tc.VAR_POSITION3D, tc.VAR_EDGES, tc.VAR_LANEPOSITION, tc.VAR_LANEPOSITION_LAT, tc.VAR_SPEED_LAT, tc.VAR_ROAD_ID, tc.VAR_ACCELERATION])
    traci.vehicle.addSubscriptionFilterLanes([-2,-1,0,1,2], noOpposite=True, downstreamDist=max_obs_range, upstreamDist=max_obs_range)

def subscribe_vehicle_ego(vehID):
    """Subscribe to store vehicle's ego information.

    Args:
        vehID (str): Vehicle ID.
    """
    traci.vehicle.subscribe(vehID, [tc.VAR_LENGTH, tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_LANE_INDEX, tc.VAR_ANGLE, tc.VAR_POSITION3D, tc.VAR_EDGES, tc.VAR_LANEPOSITION, tc.VAR_LANEPOSITION_LAT, tc.VAR_SPEED_LAT, tc.VAR_ROAD_ID, tc.VAR_ACCELERATION])    

def detected_crash():
    """Detect the crash happened in the last time step.

    Returns:
        bool: True if a collision happenes in the simulation. False if no collision happens.
    """        
    colli = traci.simulation.getCollidingVehiclesIDList()
    return colli
    
def generate_unique_numbers_with_min_difference(range_start, range_end, n, min_difference):
    if (range_end - range_start + 1) < n * (min_difference + 1):
        raise ValueError("Range is too small to generate the required number of unique numbers with the specified minimum difference.")

    # 初始化候选列表
    candidates = list(range(range_start, range_end + 1))
    result = []

    while len(result) < n:
        # 随机选择一个候选整数
        num = random.choice(candidates)
        result.append(num)

        # 移除所有与当前选择的整数差值小于 min_difference 的候选整数
        candidates = [x for x in candidates if abs(x - num) > min_difference]

    return sorted(result, reverse=False)      


def jieru_agent_gen_data(num_veh,episode_id,episode_only_crash_path):   #把训练好的agent接入到sumo中查看效果，并生成碰撞数据
    all_vehicle_step_info = {}
    all_vehs_flag_and_action_changelane = {}
    current_vehs = []
    color_red = (255, 0, 0)
    color_yellow = (255, 255, 0)
    color_blue = (0, 0, 255)
    color_green = (0, 255, 0)
    sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin/sumo-gui')
    sumo_config_file_path = r"C:\Users\28420\Desktop\SAIC\Dense-Deep-Reinforcement-Learning-main\maps\merge_400_highway\merge_400_highway.sumocfg"
    sumo_net_file_path = r"C:\Users\28420\Desktop\SAIC\Dense-Deep-Reinforcement-Learning-main\maps\merge_400_highway\merge_400_highway.net.xml"
    sumoCmd = [sumo_binary, "-c", sumo_config_file_path, "--step-length", "0.1", "--random",
               "--collision.mingap-factor", "0", "--collision.action", "warn"]
    sumoCmd += ["--lateral-resolution", "0.25"]

    # traci.start(sumoCmd, numRetries = 10)
    traci.start(sumoCmd)
    # 初始化n辆车  需要subscribe所有的车 Simulator.subscribe_vehicle_all_information
    ini_pos_num_veh = generate_unique_numbers_with_min_difference(0,400,num_veh,5)    #防止一开始就撞在一起

    for i in range(0,num_veh):
        if i == int(num_veh/3):     # 让cav位置车流1/3处
            route = f'route_{random.choice([0,1])}'
            if route == 'route_0':
                ini_lane = f'E0_{random.choice([0,1])}'
            else:
                ini_lane = 'E2_0'    
            ini_speed = random.uniform(20,40)
            # ini_position = random.uniform(0, 90)
            ini_position = ini_pos_num_veh[i]
            traci.vehicle.add('CAV', route, typeID='IDM', depart=0, departLane='first', departPos='base', departSpeed=ini_speed,
                    arrivalLane='current', arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='',
                    personCapacity=0, personNumber=0)
            traci.vehicle.moveTo('CAV', ini_lane, ini_position)
            traci.vehicle.setColor('CAV', color_red)
            traci.vehicle.setMaxSpeedLat('CAV', 4.0)
            current_vehs.append('CAV')
        else:
            route = f'route_{random.choice([0,1])}'
            if route == 'route_0':
                ini_lane = f'E0_{random.choice([0,1])}'
            else:
                ini_lane = 'E2_0'    
            ini_speed = random.uniform(20,40)
            # ini_position = random.uniform(0, 400)
            ini_position = ini_pos_num_veh[i]
            traci.vehicle.add(f'bv{i}', route, typeID='IDM', depart=0, departLane='first', departPos='base', departSpeed=ini_speed,
                    arrivalLane='current', arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='',
                    personCapacity=0, personNumber=0)
            traci.vehicle.moveTo(f'bv{i}', ini_lane, ini_position)
            traci.vehicle.setColor(f'bv{i}', color_green)
            traci.vehicle.setMaxSpeedLat(f'bv{i}', 4.0)
            current_vehs.append(f'bv{i}')

    traci.gui.trackVehicle(viewID='View #0', vehID='CAV')
    traci.gui.setZoom(viewID='View #0', zoom=500)     

    # 需要subscribe所有的车 bv Simulator.subscribe_vehicle_all_information    av
    for i in current_vehs:
        subscribe_vehicle_all_information(i)  

    for i in current_vehs:
        if i not in all_vehs_flag_and_action_changelane:
            all_vehs_flag_and_action_changelane[i] = {}
        all_vehs_flag_and_action_changelane[i]["flag_changelane"] = False
        all_vehs_flag_and_action_changelane[i]["action_changelane"] = None
        all_vehs_flag_and_action_changelane[i]["count"] = 6       

    # 记录timestep=0的所有车的观察信息
    for i in current_vehs:
        # print(_get_observation(i, 0,prev_action=None))
        if i not in all_vehicle_step_info:
            all_vehicle_step_info[i] = {}
        all_vehicle_step_info[i][0] = _get_observation(i, 0,prev_action=None)         

    time = 0.1
    # 开始模拟直到所有车辆都没了或者cav与其他车碰撞
    while True:
        veh_info = []
        time_step = 0
        all_timestep_list = []

        # # 记录每一辆车：如果采取了变道，则之后五个timestep的prev_action都为变道动作；
        # all_vehs_flag_and_action_changelane = {}
        # 记录每一辆车的prev_action
        all_vehs_prev_action = {}
        # 先得到当前有哪些车辆
        current_vehs = traci.vehicle.getIDList()
        for i in current_vehs:
            all_vehs_prev_action[i] = None

        # 被agent接管的车辆是cav周围最近的一辆车
        # 选择cav周围最近的一辆车，用agent的输出作为它的下一步动作
        choosed_veh_id = ""
        temp_length = 200
        if "CAV" in current_vehs:
            for dir in ['left', 'right']:
                temp_info = get_neighboring_following_vehicle("CAV", dir)
                if temp_info is not None and temp_info["distance"] < temp_length:
                    temp_length = temp_info["distance"]
                    choosed_veh_id = temp_info["veh_id"]
                temp_info = get_neighboring_leading_vehicle("CAV", dir)    
                if temp_info is not None and temp_info["distance"] < temp_length:
                    temp_length = temp_info["distance"]
                    choosed_veh_id = temp_info["veh_id"]
            temp_info = get_following_vehicle("CAV")
            if temp_info is not None and temp_info["distance"] < temp_length:
                temp_length = temp_info["distance"]
                choosed_veh_id = temp_info["veh_id"]
            temp_info = get_leading_vehicle("CAV")
            if temp_info is not None and temp_info["distance"] < temp_length:
                temp_length = temp_info["distance"]
                choosed_veh_id = temp_info["veh_id"]

        if choosed_veh_id != "":
            # 得到agent的obs(输入)
            agent_input = [abs(traci.vehicle.getSpeed(choosed_veh_id)-traci.vehicle.getSpeed('CAV')),temp_length]    #相对位置和相对速度
            # 得到agent的输出
            torch_discri_agent = torch_discriminator_agent(checkpoint_path=None)
            agent_output = torch_discri_agent.compute_action(agent_input)
            # print(f"agent_output: {agent_output}")
            # print(agent_output[0])
            # print(agent_output[0][0])
            agent_output = agent_output[0][0].tolist()[0]
            # print(f"agent_output: {agent_output}")
            if agent_output <= -4 :
                agent_action = {"lateral": "central", "longitudinal": agent_output}
            if agent_output >4 :
                agent_action = {"lateral": "right", "longitudinal": 0}
            if agent_output >= -4 and agent_output <= 2:
                agent_action = {"lateral": "central", "longitudinal": agent_output}
            elif agent_output > 2 and agent_output <= 3:
                agent_action = {"lateral": "left", "longitudinal": 0}
            elif agent_output >3 and agent_output <= 4:
                agent_action = {"lateral": "right", "longitudinal": 0}
            # print(f"agent_action: {agent_action}")   


            # 执行代理动作
            act(choosed_veh_id,agent_action,veh_info,time_step,all_timestep_list)  
            traci.vehicle.setColor(choosed_veh_id, color_blue)
            all_vehs_prev_action[choosed_veh_id] = agent_action   

            for i in current_vehs:
                if i != choosed_veh_id:
                    all_vehs_prev_action[i] = {"lateral": "central", "longitudinal": traci.vehicle.getAcceleration(i)}   



        
        """ 
        # 选择每辆车的动作      
        for i in current_vehs:
            is_changelane  = random.choices(['central','left','right'],weights=[0.98,0.01,0.01],k=1)[0]
            if is_changelane == 'central':
                longitudinal = random.uniform(-4,2)
            else:
                longitudinal = 0 

            # 前一个是变道且要小于等于5次变道,大于五次就不变道了
            if all_vehs_flag_and_action_changelane[i]["flag_changelane"] and all_vehs_flag_and_action_changelane[i]["count"] <= 5:
                prev_action = all_vehs_flag_and_action_changelane[i]["action_changelane"]
                all_vehs_flag_and_action_changelane[i]["count"] += 1
            else:    
                prev_action = {"lateral": is_changelane, "longitudinal": longitudinal}
                if i not in all_vehs_flag_and_action_changelane:
                    all_vehs_flag_and_action_changelane[i] = {}
                all_vehs_flag_and_action_changelane[i]["flag_changelane"] = False
                all_vehs_flag_and_action_changelane[i]["action_changelane"] = None
                all_vehs_flag_and_action_changelane[i]["count"] = 6  

            if is_changelane != "central" and all_vehs_flag_and_action_changelane[i]["count"] > 5:    
                if i not in all_vehs_flag_and_action_changelane:
                    all_vehs_flag_and_action_changelane[i] = {}
                all_vehs_flag_and_action_changelane[i]["flag_changelane"] = True
                all_vehs_flag_and_action_changelane[i]["action_changelane"] = prev_action
                all_vehs_flag_and_action_changelane[i]["count"] = 1

            act(i,prev_action,veh_info,time_step,all_timestep_list)
            all_vehs_prev_action[i] = prev_action
        """
          
        # 执行模拟当前步
        traci.simulationStep(time)
        # 记录每辆车在当前timestep的观察信息
        current_vehs = traci.vehicle.getIDList()
        for i in current_vehs:
            if i not in all_vehicle_step_info:
                all_vehicle_step_info[i] = {}
            # print('i:',all_vehs_prev_action[i])    
            all_vehicle_step_info[i][time] = _get_observation(i, time,all_vehs_prev_action[i])

        # 再判断模拟结束，如果所有车都没了，则终止模拟，没发生碰撞就不记录crash_info
        if len(current_vehs) == 0 : 
            break
        #如果cav与其他车碰撞，则终止模拟，记录当前crash_info
        collision_ids = detected_crash()
        if "CAV" in collision_ids:
            # episode_path = os.path.join(self.experiment_path, "episode_only_crash")
            if "collision_ids" not in all_vehicle_step_info:
                all_vehicle_step_info["collision_ids"] = collision_ids
            one_episode_path = os.path.join(episode_only_crash_path, f"episode_{episode_id}.json")
            with open(one_episode_path, 'w') as file:
                json.dump(all_vehicle_step_info, file,indent=4)
            break
        time +=0.1        


if __name__ == "__main__":
    for i in range(0,1):
        jieru_agent_gen_data(15,i,r"E:\pycharmcode\sumo_data_generation\merge_400_highway_with_agent_data_episode")
