"""
    This module contains the definition of several auxiliary functions designed for SUMO Reinforcement Learning.

    Authors: AAU CS (IT) 07 - 03
"""

from typing import List
import traci

def _additional_tls_info(traffic_signal):
    """ Function that initializes some useful additional attributes for the traffic signal. 
    """
    traffic_signal.links = [link[0] for link in traffic_signal.sumo.trafficlight.getControlledLinks(traffic_signal.id) if link]
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
    return [traffic_signal.sumo.edge.getLastStepHaltingNumber(edgeID) for edgeID in traffic_signal.incoming_edges]


def get_edges_avg_speed(traffic_signal) -> List[float]:
    """Returns the average speed in the last step in the incoming edges of the intersection.
    """
    return [traffic_signal.sumo.edge.getLastStepMeanSpeed(edgeID) for edgeID in traffic_signal.incoming_edges]


def get_incoming_vehicle_ids(traffic_signal) -> List[List[str]]:
    """Returns the ids of all the vehicles in the incoming edges of the intersection.
    """
    ids = []
    for edgeID in traffic_signal.incoming_edges:
        ids.append(sorted(traffic_signal.sumo.edge.getLastStepVehicleIDs(edgeID)))
    return ids


def get_crossing_vehicles(last_ids, new_ids) -> float:
    """Returns the number of vehicles from an incoming edge that have crossed the intersection, 
    computed as the number of vehicles whose id was in the incoming edge but no longer is.
    """
    crossing = 0
    i = 0
    j = 0
    while i < len(last_ids) and j < len(new_ids):
        if last_ids[i] < new_ids[j]: crossing += 1; i += 1
        elif last_ids[i] > new_ids[j]: j += 1
        else: i += 1; j += 1
    return crossing


def get_incoming_num_lanes_per_edge(traffic_signal) -> List[int]:
    """Returns the number of lanes of each of the incoming edges of the intersection.
    """
    return [traffic_signal.sumo.edge.getLaneNumber(edgeID) for edgeID in traffic_signal.incoming_edges]

