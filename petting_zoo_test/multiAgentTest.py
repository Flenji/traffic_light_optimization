# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:07:50 2023

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

env = sumo_rl.parallel_env(net_file='fumocrossing/IV/IV.net.xml',
                  route_file='fumocrossing/IV/IV.rou.xml',
                  use_gui=True,
                  num_seconds=3600,
                  observation_class = custom_observation.CustomObservationFunction,
                  reward_fn = "average-speed",
                  )


### SETTING HYPERPARAMETERS
learning_rate = 0.0025
mem_size = 1000000
eps_dec = 5e-6
batch_size = 36
gamma = 0.99
eps_min = 0.1
replace = 1000
checkpoint_dir = "results\\fourth_iteration"

### Setting the DDQN Agent for every possible agent
agents = dict.fromkeys(env.possible_agents)

for agent in agents.keys():
    input_shape = env.observation_space(agent).shape
    n_actions = env.action_space(agent).n
    name = agent + "_ddqn"
    agents[agent] = ddqn.Agent(learning_rate=learning_rate, input_dim= input_shape, n_actions=n_actions,\
                               mem_size=mem_size, eps_dec=eps_dec, eps_min = eps_min, gamma = gamma,\
                                   batch_size= batch_size, name = name, checkpoint_dir= checkpoint_dir,\
                                       replace = replace)

print(f"Agents in this simulation: {[a for a in agents.keys()]}")

observations = env.reset()[0]

while env.agents:
    actions =  {agent: agents[agent].get_action(observations[agent]) for agent in env.agents}
    
    #actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations_, rewards, terminations, truncations, infos = env.step(actions)
        
    for agent in env.agents:
        obs = observations[agent] #current observation of agent
        action = actions[agent] 
        obs_, reward, termination, truncation, info = observations_[agent],\
            rewards[agent], terminations[agent], truncations[agent], infos[agent]
            
        done = termination or truncation #TODO: see if this is needed for SUMO
        
        
        agents[agent].learn(obs, action, reward, obs_, done)
        
    observations = observations_ #setting new observation as current observation
env.close()

#observations = env.reset()[0]