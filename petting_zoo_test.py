# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 12:00:54 2023

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
from typing import List

from gymnasium import spaces
import reward_fncs
import observation_spaces

### ENVIRONMENT

env = sumo_rl.parallel_env(net_file='Networks/simple1.net.xml',
                  route_file='Networks/simple1.rou.xml',
                  #reward_fn="average-speed",
                  reward_fn=reward_fncs._combined_reward1,
                  observation_class=observation_spaces.ObservationFunction1,
                  use_gui=False,
                  num_seconds=1500)
# environment = AECEnv(env)
# environment.render_mode = "human"   
#env.env_params.additional_params.render_mode = "human"

scores = [] #utility.load_object("scores") #keeping track of scores and epsilons for vizualization
epsilons = [] #utility.load_object("epsilons")

ddqn_agent = ddqn.Agent(learning_rate = 0.00025, input_dim = (9,), n_actions = 4, \
                        mem_size = 30000, eps_dec = 1e-6, batch_size = 36, name = "ddqn", \
                            checkpoint_dir = "model_checkpoint")

#ddqn_agent.load_model() #loading a trained model

for n in range(100):    
    observations = env.reset()[0]
    print(f"Generation: {n}")
    while env.agents:
        actions = {agent: ddqn_agent.get_action(observations[agent]) for agent in env.agents}  # this is where you would insert your policy
        
        
        observations_, rewards, terminations, truncations, infos = env.step(actions)
        
        for agent in env.agents:
            obs = observations[agent] #current observation of agent
            action = actions[agent] 
            obs_, reward, termination, truncation, info = observations_[agent],\
                rewards[agent], terminations[agent], truncations[agent], infos[agent]
                
            done = termination or truncation #TODO: see if this is needed for SUMO
            
            
            ddqn_agent.learn(obs, action, reward, obs_, done)
            
        observations = observations_ #setting new observation as current observation
        scores.extend(rewards.values())
        epsilons.append(ddqn_agent.epsilon)
    
    if n % 10 == 0:
        #ddqn_agent.save_model()
        #utility.save_object(scores, "scores")
        #utility.save_object(epsilons, "epsilons")
        print(f"current epsilon: {ddqn_agent.epsilon}")

env.close()
utility.plot_learning_curve(scores, epsilons, filename = "iteration_3", path="plotting", mean_over=100)