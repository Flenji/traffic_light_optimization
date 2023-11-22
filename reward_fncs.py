"""
    This module contains the definition of several reward functions designed for SUMO Reinforcement Learning.

    Authors: AAU CS (IT) 07 - 03
"""

from typing import List

import traci


### REWARD FUNCTIONS

def _incoming_edge_congestion_reward(traffic_signal):
    """Simplest reward function, which tries to minimize the congestion in the edges aproaching the intersection.
    Congestion is evaluated exponentionally in order to penalize very congested streets."""
    congestions = traffic_signal.get_incoming_edges_density()

    return -sum(congestion^2 for congestion in congestions)
        

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
    if not hasattr(traffic_signal, 'links'):
        _additional_tls_info(traffic_signal) # Compute some extra info about the TLS

    return 0.70* traffic_signal._incoming_edge_congestion_reward() + 0.30* traffic_signal._long_waits_penalize()


### ADITIONAL FUNCTIONS

def _additional_tls_info(traffic_signal):
    """ Function that initializes some useful additional information about the traffic light system. 
    """
    traffic_signal.links = list(dict.fromkeys(traffic_signal.sumo.trafficlight.getControlledLinks(traffic_signal.id)))
    traffic_signal.incoming_edges = list(set([traci.lane.getEdgeID(link[0]) for link in traffic_signal.links]))
    traffic_signal.outgoing_edges = list(set([traci.lane.getEdgeID(link[1]) for link in traffic_signal.links]))
    
def get_incoming_edges_density(traffic_signal) -> List[float]:
    """Returns the density [0,1] of the vehicles in the incoming edges of the intersection.

    Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the edge.

    PRACTICAL CASE: The scale could be modified to [1,5] in order to admit the congestion data from the city of Barcelona as input.
    """
    edges_density = [traffic_signal.sumo.edge.getLastStepOccupancy(edgeID)/100 for edgeID in traffic_signal.incoming_edges]
    return [min(1, density) for density in edges_density]

def get_edges_queue(traffic_signal) -> List[int]:
    """Returns the number of queued vehicles of the vehicles in the incoming edges of the intersection.
    """
    edges_queued = [traffic_signal.sumo.edge.getLastStepHaltingNumber(edgeID) for edgeID in traffic_signal.incoming_edges]
    return edges_queued

###