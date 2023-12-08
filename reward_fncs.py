"""
    This module contains the definition of several reward functions designed for SUMO Reinforcement Learning.

    Authors: AAU CS (IT) 07 - 03
"""

import aux_functions


### REWARD FUNCTIONS

def _incoming_edge_congestion_reward(traffic_signal, edges):
    """Simplest reward function, which tries to minimize the congestion on the edges aproaching the intersection.
    Congestion is evaluated exponentionally in order to penalize very congested streets.
    """
    congestions = aux_functions.get_incoming_edges_density(traffic_signal, edges)

    return -sum(congestion**2 for congestion in congestions)
        

def _long_waits_penalize(traffic_signal, edges):
    """This reward function penalizes the fact that cars wait too long for a green light. The penalization is the maximum of the max waiting time of a vehicle
       on each incoming edge.
           
       In order to have the reward value between [0, 1] and be able to combine several rewards/penalizations without giving to much importance to one 
       due to the values range, we perform a kind of normalization. To do this, we assume that the maximum acceptable waiting time for a vehicle is
       95 seconds. So, we devide the sum of the max waiting times by len(traffic_signal.incoming_edges)*95. 
       (MAYBE IT IS NOT 4 BUT 2, AS TWO LANES ARE ALWAYS IN GREEN [BUT
       SOME TURNING LEFT MIGHT BE WAITING SO MAYBE IT IS CORRECT])
    """ 
    max_waits = []
        
    for edgeID in edges:
        max_time_edge = -1
        vehicles = traffic_signal.sumo.edge.getLastStepVehicleIDs(edgeID)
        for vehicleID in vehicles:
            wait = traffic_signal.sumo.vehicle.getWaitingTime(vehicleID)
            if wait > max_time_edge:
                max_time_edge = wait
        max_waits.append(min(1, max_time_edge/95))

    return -max(max_waits)


def _avg_speed_reward(traffic_signal, edges, lanes):
    """Reward function that returns the normalized sum of the average speed of the vehicles on the incoming edges of the intersection.
    
    OBS: The same speed is assumed for all the lanes of the same edge.
    """
    sum_averages = 0
    i = 0
    for edgeID in edges:
        speed = traffic_signal.sumo.edge.getLastStepMeanSpeed(edgeID)
        for laneID in lanes:
            if traffic_signal.sumo.lane.getEdgeID(laneID) == edgeID:
                sum_averages += speed/traffic_signal.sumo.lane.getMaxSpeed(laneID)
                i += 1
                break
    return sum_averages/len(edges)


def _crossing_cars_reward(traffic_signal):
    """ Reward function that returns the

    I CAN TRY TO USE THE CONGESTION ON THE OUTGOING LANES AS AN INDICATOR OF THE VEHICLES CROSSING THE INTERSECTION
    ANOTHER OPTION IS TO CALCULATE, EVERY SECOND, THE DIFFERENCE OF IDS OF EACH LANE TO KNOW THE NUMBER OF NEW ONES. 
    THEN WE WOULD NORMALIZE THE SUM BY RETURNING THE PRECENTAGE OF CARS CROSSING IN RELATION TO HOW MANY THERE WHERE IN THE EDGE

    OBS: The percentage of the total incoming vehicles that enter the intersection is used. The average of the percentage of the incoming edges could be used.
    """
    if hasattr(traffic_signal, 'last_vehicle_id'):
        crossing = 0
        total_cars = 0 
        last = traffic_signal.last_vehicle_id
        new = aux_functions.get_incoming_vehicle_ids(traffic_signal, traffic_signal.incoming_edges)
        for i in range(len(last)):
            last_ids = last[i]
            new_ids = new[i]
            total_cars += len(last_ids)
            crossing += aux_functions.get_crossing_vehicles(last_ids, new_ids)

        traffic_signal.last_vehicle_id = new
        return crossing/total_cars
    else: 
        traffic_signal.last_vehicle_id = aux_functions.get_incoming_vehicle_ids(traffic_signal, traffic_signal.incoming_edges)
        return 0


def _penalize_phase_change(traffic_signal):
    """ Reward function that penalizes the fact that the the traffic light phase is changed without a substantial benefit for the traffic. 
    Humans do not react immediately to signals and, thus, the more phase changes, the more time is lost when trying to start up the vehicle.
    """
    if hasattr(traffic_signal, 'last_phase_id') and traffic_signal.last_phase_id != traffic_signal.sumo.trafficlight.getPhase(traffic_signal.id):
        traffic_signal.last_phase_id = traffic_signal.sumo.trafficlight.getPhase(traffic_signal.id)
        return -1
    traffic_signal.last_phase_id = traffic_signal.sumo.trafficlight.getPhase(traffic_signal.id)
    return 0
    
def reward_green_to_congested(traffic_signal):
    """ Reward function that rewards the agent for letting vehicles that come from a congested edge through the intersection.

    OBS: If we want to reward the agent for clearing congested edges, it is difficult to do so by looking only at the congestion after the action as, probably,
    a lot of cars will still come to the congestion edge and, thus, the congestion level won't change even a lot of cars have been let through.
    """
    if hasattr(traffic_signal, 'last_lane_congestion'):
        current_phase = traffic_signal.sumo.trafficlight.getPhase(traffic_signal.id)
        traffic_light_program = traffic_signal.trafficlight.getCompleteRedYellowGreenDefinition(traffic_signal.id)
        total_congestion = 0
        affected_lanes = 0

        # Get the phase id of each lane and sum the congestion if the phase id is the current one.
        for phase in traffic_light_program:
            phase_id = phase._index
            lanes = traffic_signal.trafficlight.getControlledLanes(traffic_signal.id, phase_id)

        traffic_signal.last_lane_congestion = aux_functions.get_incoming_lanes_density(traffic_signal, traffic_signal.in_lanes)
        return total_congestion/affected_lanes
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
        + 0.05 * _penalize_phase_change(traffic_signal)


def _combined_reward3(traffic_signal):
    """Third reward function defined combining several factors.
    """
    edges = traffic_signal.incoming_edges
    return 0.3 * _incoming_edge_congestion_reward(traffic_signal, edges) \
        + 0.2 * _long_waits_penalize(traffic_signal, edges) \
        + 0.1 * _avg_speed_reward(traffic_signal, edges, traffic_signal.lanes) \
        + 0.35 * _crossing_cars_reward(traffic_signal) + 0.05 * _penalize_phase_change(traffic_signal)


def _combined_reward4(traffic_signal):
    """Fourth reward function defined combining several factors.
    """
    edges = traffic_signal.incoming_edges
    return 0.15 * _incoming_edge_congestion_reward(traffic_signal, edges) \
        + 0.25 * reward_green_to_congested(traffic_signal) \
        + 0.2 * _long_waits_penalize(traffic_signal, edges) \
        + 0.1 * _avg_speed_reward(traffic_signal, edges, traffic_signal.lanes) \
        + 0.25 * _crossing_cars_reward(traffic_signal) + 0.05 * _penalize_phase_change(traffic_signal)


#### MULTI-AGENT REWARD FUNCTIONS
def multi_agent_reward3(traffic_signal):
    """ First reward function for the multi-agent scenario, where traffic lights observe a wider observation space.
    Based on results of the third single-agent reward function.
    """
    direct_edges = traffic_signal.incoming_edges
    indirect_edges = traffic_signal.controlled_edges
    direct_reward = 0.3 * _incoming_edge_congestion_reward(traffic_signal, direct_edges) \
        + 0.2 * _long_waits_penalize(traffic_signal, direct_edges) \
        + 0.1 * _avg_speed_reward(traffic_signal, direct_edges, traffic_signal.direct_controlled_lanes) \
        + 0.35 * _crossing_cars_reward(traffic_signal) + 0.05 * _penalize_phase_change(traffic_signal)
    indirect_reward = 0.4 * _incoming_edge_congestion_reward(traffic_signal, indirect_edges) \
        + 0.25 * _long_waits_penalize(traffic_signal, indirect_edges) \
        + 0.35 * _avg_speed_reward(traffic_signal, indirect_edges, traffic_signal.controlled_lanes) 
        # We could add the crossing cars in the adjacent traffic lights
    
    return 0.7 * direct_reward + 0.3 * indirect_reward