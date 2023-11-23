"""
    This module contains the definition of several reward functions designed for SUMO Reinforcement Learning.

    Authors: AAU CS (IT) 07 - 03
"""

import aux_functions

### REWARD FUNCTIONS

def _incoming_edge_congestion_reward(traffic_signal):
    """Simplest reward function, which tries to minimize the congestion in the edges aproaching the intersection.
    Congestion is evaluated exponentionally in order to penalize very congested streets."""
    congestions = aux_functions.get_incoming_edges_density(traffic_signal)

    return -sum(congestion**2 for congestion in congestions)
        

def _long_waits_penalize(traffic_signal):
    """This reward function penalizes the fact that cars wait too long for a green light. The penalization is the sum of the max waiting time of a vehicle
       in each incoming edge (can it be only the max of all edges?????).
           
       In order to have the reward value between [0, 1] and be able to combine several rewards/penalizations without giving to much importance to one 
       due to the values range, we perform a kind of normalization. To do this, we assume that the maximum acceptable waiting time for a vehicle is
       95 seconds. So, we devide the sum of the max waiting times by 4*95. (MAYBE IT IS NOT 4 BUT 2, AS TWO LANES ARE ALWAYS IN GREEN [BUT
       SOME TURNING LEFT MIGHT BE WAITING SO MAYBE IT IS CORRECT])
    """ 
    sum_max_waits = 0
        
    for edgeID in traffic_signal.incoming_edges:
        max_time_edge = -1
        vehicles = traffic_signal.sumo.edge.getLastStepVehicleIDs(edgeID)
        for vehicleID in vehicles:
            wait = traffic_signal.sumo.vehicle.getWaitingTime(vehicleID)
            if wait > max_time_edge:
                max_time_edge = wait
        sum_max_waits += max_time_edge

    return -min(1, sum_max_waits/(4*95))
    
def _combined_reward1(traffic_signal):
    """First reward function defined combining several factors.
    """
    return 0.70* _incoming_edge_congestion_reward(traffic_signal) + 0.30* _long_waits_penalize(traffic_signal)