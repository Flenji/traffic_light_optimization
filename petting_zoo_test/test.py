# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 17:08:06 2023

@author: hanne
"""

import sumo_rl
from pettingzoo import AECEnv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import ddqn
import utility
import os
import custom_observation
import sumolib.net as net

net_file = 'fumos/2x2/2x2.net.xml'
custom_observation.ComplexObservationFunction.net_file = net_file
custom_observation.ComplexObservationFunction.radius = 2
x = custom_observation.ComplexObservationFunction 

env = sumo_rl.parallel_env(net_file='fumos/2x2/2x2.net.xml',
                  route_file='fumos/2x2/2x2.rou.xml',
                  use_gui=True,
                  num_seconds=1500,
                  observation_class = custom_observation.ComplexObservationFunction,
                  reward_fn = "average-speed",
                  )

obs = env.reset() 

obj = x.compObject

net = obj.net

traffic_light = "D1"#obj.connected_traffic_lights[-1]

lanes = list(dict.fromkeys(obj.sumo.trafficlight.getControlledLanes(obj.ts.id)))


out_going = []
for lane in lanes:
    lane_object = net.getLane(lane)
    outgoingLanesObj = lane_object.getOutgoingLanes()
    
    outgoingLanes = [outgoing.getID() for outgoing in outgoingLanesObj]
    
    out_going.extend(outgoingLanes)

out_going = list(dict.fromkeys(out_going))

con_lanes_list = [out_going] 
for i in range(2):
    next_lanes = [] 
    for lane in con_lanes_list[-1]: #for lane in the list of last calculated lanes
        lane_object = net.getLane(lane)
        next_lanes.extend( [l.getID() for l in lane_object.getOutgoingLanes()])
    con_lanes_list.append(next_lanes)
    
controlled_lanes = list({l for sublist in con_lanes_list for l in sublist}) #to remove duplicates

def step():
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy
    return env.step(actions)[0]
    

def getSameLanes(xs,ys):
    return [x for x in xs if x in ys]


edge_ids = []

for lane in controlled_lanes:
    lane_object = net.getLane(lane)
    edge_object = lane_object.getEdge()
    edge_ids.append(edge_object.getID())

edge_ids = sorted(dict.fromkeys(edge_ids))


edges_length = {}
for edge in edge_ids:
    edge_object = net.getEdge(edge)
    edges_length[edge] = edge_object.getLength()
    

edges_density = [
    obj.sumo.edge.getLastStepVehicleNumber(edge)
    / ((edges_length[edge]) /(obj.ts.MIN_GAP + obj.sumo.edge.getLastStepLength(edge)))
    for edge in edge_ids
    ]

dens = [min (1, density) for density in edges_density]


"""
sumo_env = env.unwrapped.env.sumo
traffic_light = sumo_env.trafficlight.getIDList()[0]

controlled_lanes = list(dict.fromkeys( sumo_env.trafficlight.getControlledLanes(traffic_light))) #get the lanes that are directly connected to the traffic_light


def getControlledLanesRadius(controlled_lanes,radius):
    con_lanes_list = [controlled_lanes] 
    for i in range(radius):
        next_lanes = [] 
        for lane in con_lanes_list[-1]: #for lane in the list of last calculated lanes
            lane_object = net.getLane(lane)
            next_lanes.extend( [l.getID() for l in lane_object.getIncoming()])
        con_lanes_list.append(next_lanes)
        
    controlled_lanes = list({l for sublist in con_lanes_list for l in sublist}) #to remove duplicates
        
    return controlled_lanes

controlled_lanes = getControlledLanesRadius(controlled_lanes, 1)

#check whicht traffic lights should be added to the ovsservationspace 
connected_traffic_lights = []
traffic_lights = [tl.getID() for tl in net.getTrafficLights()]
for tl in traffic_lights:
    
    tl_controlled_lanes = sumo_env.trafficlight.getControlledLanes(tl)
    for lane in tl_controlled_lanes:
        if lane in controlled_lanes:
            print(tl)
            print("traffic_light connected")
            break



    
print(f"{controlled_lanes}")

phases = sumo_env.trafficlight.getAllProgramLogics("C1")[0].phases
green_phases = []
num_green_phases = 0

for phase in phases:
    state = phase.state
    if "y" not in state and (state.count("r") + state.count("s") != len(state)):
        num_green_phases += 1
        green_phases.append(state)

current_state = sumo_env.trafficlight.getRedYellowGreenState("C1")
phase_id = [1  if current_state == state else 0 for state in green_phases] #one-hot encoding



    
#env.close()

"""
