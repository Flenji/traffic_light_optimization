"""
    This module contains the definition of several reward functions designed for SUMO Reinforcement Learning.

    Authors: AAU CS (IT) 07 - 03
"""

import aux_functions
import traci


### REWARD FUNCTIONS

def _incoming_edge_congestion_reward(traffic_signal, edges):
    """Simplest reward function, which tries to minimize the congestion on the edges aproaching the intersection.
    Congestion is evaluated exponentionally in order to penalize very congested streets.
    """
    congestions = aux_functions.get_incoming_edges_density(traffic_signal, edges)
    #print("_incoming_edge_congestion_reward")
    #print("congestions: ", congestions)
    return 1-(sum(congestion**2 for congestion in congestions)/len(edges))
        

def _long_waits_penalize(traffic_signal, edges):
    """This reward function penalizes the fact that cars wait too long for a green light. The penalization is the maximum of the max waiting time of a vehicle
       on each incoming edge.
           
       In order to have the reward value between [0, 1] and be able to combine several rewards/penalizations without giving to much importance to one 
       due to the values range, we perform a kind of normalization. To do this, we assume that the maximum acceptable waiting time for a vehicle is
       95 seconds. So, we devide the max waiting times by 95.
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
    """Reward function that returns the normalized sum of the average speed of the vehicles on the incoming edges of the intersection.
    
    OBS: The same speed is assumed for all the lanes of the same edge.
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
    """ Reward function that returns the

    I CAN TRY TO USE THE CONGESTION ON THE OUTGOING LANES AS AN INDICATOR OF THE VEHICLES CROSSING THE INTERSECTION
    ANOTHER OPTION IS TO CALCULATE, EVERY SECOND, THE DIFFERENCE OF IDS OF EACH LANE TO KNOW THE NUMBER OF NEW ONES. 
    THEN WE WOULD NORMALIZE THE SUM BY RETURNING THE PRECENTAGE OF CARS CROSSING IN RELATION TO HOW MANY THERE WHERE IN THE EDGE

    OBS: The percentage of the total incoming vehicles that enter the intersection is used. The average of the percentage of the incoming edges could be used.

    It creates the attribute for the traffic_signal "green_lanes", which is needed in other reward functions.
    """
    if (hasattr(traffic_signal, 'last_direct_vehicle_id') and direct) or (hasattr(traffic_signal, 'last_global_vehicle_id') and not direct):
        crossing = 0
        total_cars = 0
        if direct:
            last = traffic_signal.last_direct_vehicle_id
            green_lanes = [False for i in range(len(last))] # Vector used to indicate the lanes which have some vehicles crossing -> have green light 
        else: 
            last = traffic_signal.last_global_vehicle_id
        new = aux_functions.get_incoming_vehicle_ids(traffic_signal, lanes)
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
        traffic_signal.last_direct_vehicle_id = aux_functions.get_incoming_vehicle_ids(traffic_signal, lanes)
    else: 
        traffic_signal.last_global_vehicle_id = aux_functions.get_incoming_vehicle_ids(traffic_signal, lanes)
    return 0


def _penalize_phase_change(traffic_signal):
    """ Reward function that penalizes the fact that the the traffic light phase is changed without a substantial benefit for the traffic. 
    Humans do not react immediately to signals and, thus, the more phase changes, the more time is lost when trying to start up the vehicle.
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
    

def _BROKEN_penalize_phase_change(traffic_signal):
    """ Proper version of the reward function that penalizes phase changes but, 
    presumably, without any effect as getPhase() seems to return always the same index. 
    """
    new_phase = traci.trafficlight.getPhase(traffic_signal.id)
    if hasattr(traffic_signal, 'last_phase') and traffic_signal.last_phase != new_phase:
        traffic_signal.last_phase = new_phase
        return 0
    traffic_signal.last_phase = new_phase
    return 1
 

def _reward_green_to_congested(traffic_signal):
    """ Reward function that rewards the agent for letting vehicles that come from a congested edge through the intersection.

    OBS: If we want to reward the agent for clearing congested edges, it is difficult to do so by looking only at the congestion after the action as, probably,
    a lot of cars will still come to the congestion edge and, thus, the congestion level won't change even a lot of cars have been let through.
    """
    #print("reward_green_to_congested")
    if hasattr(traffic_signal, 'last_lane_congestion'):
        last_congestions = traffic_signal.last_lane_congestion
        total_congestion = 0
        affected_lanes = 0

        green_lanes = traffic_signal.green_lanes
        for i in range(len(green_lanes)):
            if green_lanes[i]:
                affected_lanes += 1
                total_congestion += last_congestions[i]

        traffic_signal.last_lane_congestion = aux_functions.get_incoming_lanes_density(traffic_signal, traffic_signal.in_lanes)
        #print("Total congestion: ", total_congestion, " Affected lanes: ", affected_lanes)

        if affected_lanes != 0 and total_congestion/affected_lanes > 1:
            print("reward_green", last_congestions)

        return total_congestion/affected_lanes if affected_lanes != 0 else 0
    traffic_signal.last_lane_congestion = aux_functions.get_incoming_lanes_density(traffic_signal, traffic_signal.in_lanes)
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
        + 0.05 * _BROKEN_penalize_phase_change(traffic_signal)


def _combined_reward3(traffic_signal):
    """Third reward function defined combining several factors.
    """
    #print(traci.trafficlight.getPhase(traffic_signal.id))
    edges = traffic_signal.incoming_edges
    return 0.3 * _incoming_edge_congestion_reward(traffic_signal, edges) \
        + 0.2 * _long_waits_penalize(traffic_signal, edges) \
        + 0.1 * _avg_speed_reward(traffic_signal, edges, traffic_signal.in_lanes) \
        + 0.35 * _crossing_cars_reward(traffic_signal, traffic_signal.in_lanes, True) + 0.05 * _BROKEN_penalize_phase_change(traffic_signal)


def _combined_reward4(traffic_signal):
    """Fourth reward function defined combining several factors.
    """
    edges = traffic_signal.incoming_edges
    return 0.15 * _incoming_edge_congestion_reward(traffic_signal, edges) \
        + 0.2 * _long_waits_penalize(traffic_signal, edges) \
        + 0.1 * _avg_speed_reward(traffic_signal, edges, traffic_signal.in_lanes) \
        + 0.25 * _crossing_cars_reward(traffic_signal, traffic_signal.in_lanes, True) \
        + 0.25 * _reward_green_to_congested(traffic_signal) + 0.05 * _BROKEN_penalize_phase_change(traffic_signal)


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
    global_reward = 0.4 * _incoming_edge_congestion_reward(traffic_signal, global_edges) \
        + 0.25 * _long_waits_penalize(traffic_signal, global_edges) \
        + 0.35 * _avg_speed_reward(traffic_signal, global_edges, traffic_signal.controlled_lanes) 
    
    return 0.7 * direct_reward + 0.3 * global_reward


def multi_agent_reward3_2(traffic_signal):
    """ First reward function for the multi-agent scenario, where traffic lights observe a wider observation space.
    Based on the study of the third single-agent reward function.

    The crossing cars for the global observed lanes are rewarded too.
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