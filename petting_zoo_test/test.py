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

net = net.readNet('fumocrossing/second.net.xml')

env = sumo_rl.parallel_env(net_file='fumocrossing/second.net.xml',
                  route_file='fumocrossing/second.rou.xml',
                  use_gui=True,
                  num_seconds=1500,
                  observation_class = custom_observation.CustomObservationFunction,
                  reward_fn = "average-speed",
                  )

obs = env.reset() 

actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy

sumo_env = env.unwrapped.env.sumo
traffic_light = sumo_env.trafficlight.getIDList()[0]

controlled_lanes = list(dict.fromkeys( sumo_env.trafficlight.getControlledLanes(traffic_light))) #get the lanes that are directly connected to the traffic_light


def getControlledLanesRadius(controlled_lanes,radius):
    con_lanes_list = [controlled_lanes]
    for i in range(radius):
        next_lanes = []
        for lane in con_lanes_list[-1]:
            lane_object = net.getLane(lane)
            next_lanes.extend( [l.getID() for l in lane_object.getIncoming()])
            #next_lanes = sumo_env.lane.getLinks(lane, extended = False)#[0]#[:2]
        con_lanes_list.append(next_lanes)
        
        controlled_lanes = list({l for sublist in con_lanes_list for l in sublist})
        
        return controlled_lanes

controlled_lanes = getControlledLanesRadius(controlled_lanes, 1)

traffic_lights = [tl.getID() for tl in net.getTrafficLights()]
for tl in traffic_lights:
    
    tl_controlled_lanes = sumo_env.trafficlight.getControlledLanes(tl)
    for lane in tl_controlled_lanes:
        if lane in controlled_lanes:
            print(lane)
            print("traffic_light connected")
            break


    
print(f"{controlled_lanes}")
    
#env.close()


