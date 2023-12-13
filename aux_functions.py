"""
    This module contains the definition of several auxiliary functions designed for SUMO Reinforcement Learning agent training.

    Authors: AAU CS (IT) 07 - 03
"""

from typing import List
import traci

def _additional_tls_info(traffic_signal):
    """ Function that initializes some useful additional attributes for the traffic signal. 
    """
    traffic_signal.links = [link[0] for link in traci.trafficlight.getControlledLinks(traffic_signal.id) if link]
    traffic_signal.in_lanes = list(set([link[0] for link in traffic_signal.links]))
    traffic_signal.out_lanes = list(set([link[1] for link in traffic_signal.links]))
    traffic_signal.incoming_edges = list(set([traci.lane.getEdgeID(lane) for lane in traffic_signal.in_lanes]))
    traffic_signal.outgoing_edges = list(set([traci.lane.getEdgeID(lane) for lane in traffic_signal.out_lanes]))


def get_edges_density(traffic_signal, edges) -> List[float]:
    """Returns the density [0,1] of the vehicles in the incoming edges of the intersection.

    Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the edge.
    """
    edges_density = [traci.edge.getLastStepOccupancy(edgeID) for edgeID in edges]
    return [min(1, density) for density in edges_density]


def get_lanes_density(traffic_signal, lanes) -> List[float]:
    """Returns the density [0,1] of the vehicles in the incoming lanes of the intersection.

    Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
    """
    return [traci.lane.getLastStepOccupancy(laneID) for laneID in lanes]


def get_edges_queue(traffic_signal, edges) -> List[int]:
    """Returns the number of queued vehicles of the vehicles in the incoming edges of the intersection.
    """
    return [traci.edge.getLastStepHaltingNumber(edgeID) for edgeID in edges]

def get_lanes_queue(traffic_signal, lanes) -> List[int]:
    """Returns the number of queued vehicles of the vehicles in the incoming lanes of the intersection.
    """
    return [traci.lane.getLastStepHaltingNumber(laneID) for laneID in lanes]


def get_edges_avg_speed(traffic_signal, edges) -> List[float]:
    """Returns the average speed in the last step in the incoming edges of the intersection.
    """
    return [traci.edge.getLastStepMeanSpeed(edgeID) for edgeID in edges]


def get_vehicle_ids(traffic_signal, lanes) -> List[List[str]]:
    """Returns the ids of all the vehicles in the incoming lanes of the intersection.
    """
    ids = []
    for laneID in lanes:
        ids.append(sorted(traci.lane.getLastStepVehicleIDs(laneID)))
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


def get_num_lanes_per_edge(traffic_signal, edges) -> List[int]:
    """Returns the number of lanes of each of the given edges.
    """
    return [traci.edge.getLaneNumber(edgeID) for edgeID in edges]


def getPhases(traffic_signal):
    """
    Creates the list of every possible green phase configuration for the traffic light.
    """
    phases = traci.trafficlight.getAllProgramLogics(traffic_signal.id)[0].phases
    green_phases = []    
    for phase in phases:
        state = phase.state
        if "y" not in state and (state.count("r") + state.count("s") != len(state)):
            green_phases.append(state)
    traffic_signal.tl_green_phases = green_phases
