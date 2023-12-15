"""
    This module contains the definition of several reward functions designed for SUMO Reinforcement Learning agent training.

    Authors: AAU CS (IT) 07 - 03
"""

import aux_functions
import traci


### REWARD FUNCTIONS

def _incoming_edge_congestion_reward(traffic_signal, edges):
    """Reward function that tries to minimize the congestion on the edges aproaching the intersection.

        Congestion is evaluated exponentionally in order to penalize very congested streets.

        RETURN: One - Average of the squared congestion percentage of the incoming edges. RANGE: [0, 1]
    """
    congestions = aux_functions.get_edges_density(edges)
    #print("_incoming_edge_congestion_reward")
    #print("congestions: ", congestions)
    return 1-(sum(congestion**2 for congestion in congestions)/len(edges))
        

def _long_waits_penalize(traffic_signal, edges):
    """Reward function that penalizes the fact that cars wait too long for a green light. 
           
       OBS: In order to have the reward value between [0, 1] and be able to combine several rewards/penalizations without giving to much importance to one 
       due to the values' range, we perform a kind of normalization. To do this, we assume that the maximum acceptable waiting time for a vehicle is
       95 seconds. So, we divide the max waiting times by 95 and get the minimum between that ot 1.

       RETURN: One - Maximum of the max waiting time of a vehicle on each incoming edge. RANGE: [0, 1]
    """ 
    max_waits = []
        
    for edgeID in edges:
        max_time_edge = -1
        vehicles = traci.edge.getLastStepVehicleIDs(edgeID)
        for vehicleID in vehicles:
            wait = traci.vehicle.getWaitingTime(vehicleID)
            if wait > max_time_edge:
                max_time_edge = wait
        max_waits.append(min(1, max_time_edge/95))
    #print("_long_waits_penalize")
    #print("waiting times: ", max_waits)
    return 1-max(max_waits)


def _avg_speed_reward(traffic_signal, edges, lanes):
    """Reward function that rewards the fact that cars are moving and, additionally, moving faster.
    
       OBS: The same speed is assumed for all the lanes of the same edge.

       RETURN: Average of the average normalized speed on each incoming edge. RANGE: [0, 1]
    """
    sum_averages = 0
    i = 0
    for edgeID in edges:
        speed = traci.edge.getLastStepMeanSpeed(edgeID)
        for laneID in lanes:
            if traci.lane.getEdgeID(laneID) == edgeID:
                sum_averages += speed/traci.lane.getMaxSpeed(laneID)
                i += 1
                break
    #print("_avg_speed_reward")
    #print("sum averages speed: ", sum_averages)
    return sum_averages/len(edges)


def _crossing_cars_reward(traffic_signal, lanes, direct):
    """ Reward function that rewards the agent for letting cars through the intersection.

    OBS: The percentage of the total incoming vehicles that enter the intersection is used. 
         The average of the percentage of the incoming edges/lanes could be used.

         It creates the attribute for the traffic_signal "green_lanes", which is needed in other reward functions.

    RETURN: Percentage of cars not present anymore (Difference of IDs of each lane between the last and the current step) 
            among all the ones there were approaching the intersection in the previous step. RANGE: [0, 1]
    """
    if (hasattr(traffic_signal, 'last_direct_vehicle_id') and direct) or (hasattr(traffic_signal, 'last_global_vehicle_id') and not direct):
        crossing = 0
        total_cars = 0
        if direct:
            last = traffic_signal.last_direct_vehicle_id
            green_lanes = [False for i in range(len(last))] # Vector used to indicate the lanes which have some vehicles crossing -> have green light 
        else: 
            last = traffic_signal.last_global_vehicle_id
        new = aux_functions.get_vehicle_ids(lanes)
        for i in range(len(last)):
            last_ids = last[i]
            new_ids = new[i]
            total_cars += len(last_ids)
            lane_crossing = aux_functions.get_crossing_vehicles(last_ids, new_ids)
            crossing += lane_crossing
            if direct and lane_crossing > 0: # Some vehicles from lane i crossed the intersection
                green_lanes[i] = True
        if direct:
            traffic_signal.last_direct_vehicle_id = new
            traffic_signal.green_lanes = green_lanes
        else: 
            traffic_signal.last_global_vehicle_id = new
        #print("_crossing_cars_reward")
        #print("Crossing cars: ", crossing, " Total: ", total_cars)

        if total_cars != 0 and crossing/total_cars > 1:
            print("crossing", crossing/total_cars)

        return crossing/total_cars if total_cars != 0 else 0
    if direct:
        traffic_signal.last_direct_vehicle_id = aux_functions.get_vehicle_ids(lanes)
    else: 
        traffic_signal.last_global_vehicle_id = aux_functions.get_vehicle_ids(lanes)
    return 0


def _penalize_phase_change(traffic_signal):
    """ Reward function that penalizes the fact that the the traffic light phase is changed without a substantial benefit for the traffic. 
    Humans do not react immediately to signals and, thus, the more phase changes, the more time lost when trying to start up the vehicle.

    RETURN: 0 if there has been a phase change, 1 if there has not. RANGE: {0, 1}
    """
    if not hasattr(traffic_signal, 'tl_green_phases'):
        aux_functions.getPhases(traffic_signal)
    current_state = traci.trafficlight.getRedYellowGreenState(traffic_signal.id)
    phase_id = [1  if current_state == state else 0 for state in traffic_signal.tl_green_phases]
    if hasattr(traffic_signal, 'last_phase') and traffic_signal.last_phase != phase_id:
        traffic_signal.last_phase = phase_id
        return 0
    traffic_signal.last_phase = phase_id
    return 1
    

def _reward_green_to_congested(traffic_signal, lanes):
    """ Reward function that rewards the agent for letting vehicles that come from a congested lane through the intersection.

    OBS: If we want to reward the agent for clearing congested lanes, it is difficult to do so by looking only at the congestion after the action as, probably,
    a lot of cars will still come to the congested lane and, thus, the congestion level won't change even if a lot of cars have been let through.

    RETURN: Sum of the congestions, had by the lanes that have green light, before the simulation step. RANGE: [0,1]
    """
    #print("reward_green_to_congested")
    if hasattr(traffic_signal, 'last_lane_congestion'):
        last_congestions = traffic_signal.last_lane_congestion
        #print(last_congestions)
        total_congestion = 0

        green_lanes = traffic_signal.green_lanes
        for i in range(len(green_lanes)):
            if green_lanes[i]:
                total_congestion += last_congestions[i]

        traffic_signal.last_lane_congestion = aux_functions.get_lanes_density(lanes)
        #print(total_congestion)
        return min(1, total_congestion)
    traffic_signal.last_lane_congestion = aux_functions.get_lanes_density(lanes)
    return 0


#### COMBINED REWARD FUNCTIONS ####
    
def _combined_reward1(traffic_signal):
    """First reward function defined combining several factors.
    """
    edges = traffic_signal.incoming_edges
    return 0.85 * _incoming_edge_congestion_reward(traffic_signal, edges) \
        + 0.15 * _long_waits_penalize(traffic_signal, edges)


def _combined_reward2(traffic_signal):
    """Second reward function defined combining several factors.
    """
    edges = traffic_signal.incoming_edges
    return 0.8 * _incoming_edge_congestion_reward(traffic_signal, edges) \
        + 0.15 * _long_waits_penalize(traffic_signal, edges) \
        + 0.05 * _penalize_phase_change(traffic_signal)


def _combined_reward3(traffic_signal):
    """Third reward function defined combining several factors.
    """
    #print(traci.trafficlight.getPhase(traffic_signal.id))
    edges = traffic_signal.incoming_edges
    return 0.3 * _incoming_edge_congestion_reward(traffic_signal, edges) \
        + 0.2 * _long_waits_penalize(traffic_signal, edges) \
        + 0.1 * _avg_speed_reward(traffic_signal, edges, traffic_signal.in_lanes) \
        + 0.35 * _crossing_cars_reward(traffic_signal, traffic_signal.in_lanes, True) + 0.05 * _penalize_phase_change(traffic_signal)


def _combined_reward4(traffic_signal):
    """Fourth reward function defined combining several factors.
    """
    edges = traffic_signal.incoming_edges
    return 0.15 * _incoming_edge_congestion_reward(traffic_signal, edges) \
        + 0.2 * _long_waits_penalize(traffic_signal, edges) \
        + 0.1 * _avg_speed_reward(traffic_signal, edges, traffic_signal.in_lanes) \
        + 0.25 * _crossing_cars_reward(traffic_signal, traffic_signal.in_lanes, True) \
        + 0.25 * _reward_green_to_congested(traffic_signal, traffic_signal.in_lanes) + 0.05 * _penalize_phase_change(traffic_signal)


#### MULTI-AGENT REWARD FUNCTIONS

def multi_agent_reward3(traffic_signal):
    """ First reward function for the multi-agent scenario, where traffic lights observe a wider observation space.
    Based on the study of the third single-agent reward function.
    """
    direct_edges = traffic_signal.incoming_edges
    global_edges = traffic_signal.controlled_edges
    direct_reward = 0.3 * _incoming_edge_congestion_reward(traffic_signal, direct_edges) \
        + 0.2 * _long_waits_penalize(traffic_signal, direct_edges) \
        + 0.1 * _avg_speed_reward(traffic_signal, direct_edges, traffic_signal.direct_controlled_lanes) \
        + 0.35 * _crossing_cars_reward(traffic_signal, traffic_signal.direct_controlled_lanes, True) + 0.05 * _penalize_phase_change(traffic_signal)
    global_reward = 0.3 * _incoming_edge_congestion_reward(traffic_signal, global_edges) \
        + 0.2 * _long_waits_penalize(traffic_signal, global_edges) \
        + 0.15 * _avg_speed_reward(traffic_signal, global_edges, traffic_signal.controlled_lanes)  \
        + 0.35 * _crossing_cars_reward(traffic_signal, traffic_signal.controlled_lanes, False)
    
    return 0.7 * direct_reward + 0.3 * global_reward


def multi_agent_reward4(traffic_signal):
    """ Third reward function for the multi-agent scenario, where traffic lights observe a wider observation space.
    Based on the study of the fourth single-agent reward function.
    """
    direct_edges = traffic_signal.incoming_edges
    global_edges = traffic_signal.controlled_edges
    direct_reward = 0.15 * _incoming_edge_congestion_reward(traffic_signal, direct_edges) \
        + 0.2 * _long_waits_penalize(traffic_signal, direct_edges) \
        + 0.1 * _avg_speed_reward(traffic_signal, direct_edges, traffic_signal.direct_controlled_lanes) \
        + 0.25 * _crossing_cars_reward(traffic_signal, traffic_signal.direct_controlled_lanes, True) \
        + 0.25 * _reward_green_to_congested(traffic_signal, traffic_signal.controlled_lanes) + 0.05 * _penalize_phase_change(traffic_signal)
    global_reward = 0.3 * _incoming_edge_congestion_reward(traffic_signal, global_edges) \
        + 0.2 * _long_waits_penalize(traffic_signal, global_edges) \
        + 0.15 * _avg_speed_reward(traffic_signal, global_edges, traffic_signal.controlled_lanes)  \
        + 0.35 * _crossing_cars_reward(traffic_signal, traffic_signal.controlled_lanes, False)
    
    return 0.7 * direct_reward + 0.3 * global_reward