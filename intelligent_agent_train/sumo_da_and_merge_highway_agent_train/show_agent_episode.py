import json
import os, sys
import shutil
import traci
import traci.constants as tc
import sumolib
import numpy as np
import math

def show_crash_episode_new(crash_vehicles_all_times = ""):
    color_red = (255,0,0)
    color_yellow = (255,255,0)
    color_blue = (0,0,255)
    color_green = (0,255,0)
    lc_duration = 1
    step_size = 0.1
    action_step_size = 0.1
    route_0 = ["E0_0","E0_1","E1_0","E1_1"]
    route_1 = ["E2_0","E1_0","E1_1"]

    sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin/sumo-gui')
    sumo_config_file_path = r"C:\Users\28420\Desktop\SAIC\Dense-Deep-Reinforcement-Learning-main\maps\merge_400_highway\merge_400_highway.sumocfg"
    sumo_net_file_path = r"C:\Users\28420\Desktop\SAIC\Dense-Deep-Reinforcement-Learning-main\maps\merge_400_highway\merge_400_highway.net.xml"
    sumoCmd = [sumo_binary, "-c", sumo_config_file_path, "--step-length", str(step_size), "--random",
               "--collision.mingap-factor", "0", "--collision.action", "warn"]
    sumoCmd += ["--lateral-resolution", "0.25"]

    # 记录所有timesteps
    all_timestep_list = []
    # 维护当前车辆集合
    current_vehicles_id = []
    with open(crash_vehicles_all_times, 'r') as f:
        alldata = json.load(f)
    current_vehicles_id = list(alldata.keys())
    current_vehicles_id.remove("collision_ids")
    # print(current_vehicles_id)
    temp_timestep_list = list(alldata["CAV"].keys())
    num_timesteps = 0
    # 得到 最长时间戳序列 及 它的长度
    for i in current_vehicles_id:
        temp_timestep_list = list(alldata[i].keys())
        if len(temp_timestep_list)>num_timesteps:
            num_timesteps = len(temp_timestep_list)
            all_timestep_list = temp_timestep_list
    # print(all_timestep_list)
    # print(num_timesteps)
    # print(current_vehicles_id)
    traci.start(sumoCmd, numRetries = 10)

    # 初始化添加所有的车辆
    for i in current_vehicles_id:
        if i == "CAV":
            road_id = alldata["CAV"]["0"]["Ego"]["road_id"]
            lane_index = alldata["CAV"]["0"]["Ego"]["lane_index"]
            init_speed = alldata[i]["0"]["Ego"]["velocity"] 
            init_position = math.sqrt(alldata[i]["0"]["Ego"]["position"][0]**2+alldata[i]["0"]["Ego"]["position"][1]**2)
            print(init_position)
            if f'{road_id}_{lane_index}' in route_0:
                ini_route = 'route_0'
            else:
                ini_route = 'route_1'
            traci.vehicle.add('CAV', ini_route, typeID='IDM', depart=None, departLane='first', departPos='base', departSpeed=init_speed, arrivalLane='current', arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=0, personNumber=0)
            traci.vehicle.moveTo('CAV', f'{road_id}_{lane_index}', init_position)
            # movetoxy载入不进去cav位置
            # traci.vehicle.moveToXY('CAV', road_id, lane_index,alldata[i]["0"]["Ego"]["position"][0],alldata[i]["0"]["Ego"]["position"][1])
            traci.vehicle.setColor('CAV', color_red)
            traci.vehicle.setMaxSpeedLat('CAV', 4.0)
            print(-1,traci.vehicle.getPosition("CAV"))
            traci.gui.trackVehicle(viewID='View #0', vehID='CAV')
            traci.gui.setZoom(viewID='View #0', zoom=500)
        else:
            road_id = alldata[i]["0"]["Ego"]["road_id"]
            lane_index = alldata[i]["0"]["Ego"]["lane_index"]
            init_speed = alldata[i]["0"]["Ego"]["velocity"]
            init_position = math.sqrt(alldata[i]["0"]["Ego"]["position"][0]**2+alldata[i]["0"]["Ego"]["position"][1]**2)
            if f'{road_id}_{lane_index}' in route_0:
                ini_route = 'route_0'
            else:
                ini_route = 'route_1'
            traci.vehicle.add(i, ini_route, typeID='IDM', depart=None, departLane='first', departPos='base', departSpeed=init_speed, arrivalLane='current', arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=0, personNumber=0)
            traci.vehicle.moveTo(i, f'{road_id}_{lane_index}', init_position)
            if i in alldata["collision_ids"]:
                traci.vehicle.setColor(i, color_blue)
            else:
                traci.vehicle.setColor(i, color_green)
            traci.vehicle.setMaxSpeedLat(i, 4.0)
    # print(traci.vehicle.getSpeed("CAV") )    
    # traci.vehicle.setAccel("CAV","0.1") 
    # traci.simulationStep(0)
    # print(traci.vehicle.getSpeed("CAV") )
    print(0,traci.vehicle.getPosition("CAV"))
    i=1
    # 开始循环 决定每一辆车下一步的动作 并且 traci.simulationStep(time)
    while True :
        
        # print(traci.vehicle.getIDList())
        # 决定每一辆车的动作
        for veh_id in current_vehicles_id:
            if i >= len(alldata[veh_id]):   #如果当前的timestep大于当前车辆的生存周期，则不更新它的动作
                # print(i,veh_id)
                continue
            else:
                cur_prev_action = alldata[veh_id][all_timestep_list[i]]["Ego"]["prev_action"]
                if cur_prev_action == None:
                    cur_prev_action = {"lateral": "central", "longitudinal": alldata[veh_id][all_timestep_list[i]]["Ego"]["acceleration"]}
                elif cur_prev_action["lateral"] == "central" :
                    cur_prev_action = {"lateral": "central", "longitudinal": alldata[veh_id][all_timestep_list[i]]["Ego"]["acceleration"]}    
                # print(veh_id)     
                act(veh_id,cur_prev_action,alldata[veh_id],i,all_timestep_list)
        traci.simulationStep(all_timestep_list[i])       
        print(i,traci.vehicle.getPosition("CAV"))
        if i == len(all_timestep_list) :
            break    
        i+=1 
    #     pass

# veh_info 是一辆车从头到尾的信息
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
    # print("current_velocity ",current_velocity," controlled_acc ",controlled_acc)
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

def change_vehicle_speed(vehID, acceleration, duration=1.0,step_size=0.1):
    init_speed = traci.vehicle.getSpeed(vehID)
    final_speed = init_speed+acceleration*(step_size+duration)
    if final_speed < 0:
        final_speed = 0
    traci.vehicle.slowDown(vehID, final_speed, duration)

def change_vehicle_lane(vehID, direction, duration=1.0, sublane_flag=True,step_size=0.1):
    if sublane_flag:
        latdist = _cal_lateral_distance(vehID, direction)
        lat_max_v = _cal_lateral_maxSpeed(vehID, abs(latdist), duration)
        traci.vehicle.setMaxSpeedLat(vehID, lat_max_v)
        traci.vehicle.changeSublane(vehID, latdist)
    else:
        if direction == "left":
            indexOffset = 1
        elif direction == "right":
            indexOffset = -1
        else:
            raise ValueError("Unknown direction for lane change command")
        traci.vehicle.changeLaneRelative(vehID, indexOffset, step_size)


def get_available_lanes(sumo_net_file_path,edge_id=None):
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
    b, c = lat_acc*(time), lat_acc*lane_width
    delta_power = b**2-4*c
    if delta_power >= 0:
        lat_max_v = (-math.sqrt(delta_power)+b)/2
    else:
        raise ValueError("The lateral maximum acceleration is too small.")
    return lat_max_v

def _cal_lateral_distance(vehID, direction):
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


if __name__ == "__main__":
    show_crash_episode_new('E:\pycharmcode\sumo_data_generation\merge_400_highway_with_agent_data_episode\episode_24.json')