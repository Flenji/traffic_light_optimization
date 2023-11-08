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
import custom_observation 

env = sumo_rl.parallel_env(net_file='fumocrossing/fumocrossing.net.xml',
                  route_file='fumocrossing/fumocrossing.rou.xml',
                  use_gui=True,
                  num_seconds=3000,
                  observation_class = custom_observation.CustomObservationFunction,
                  reward_fn = "average-speed")
# environment = AECEnv(env)
# environment.render_mode = "human"   
# env.env_params.additional_params.render_mode = "human"


scores = [] #keeping track of scores and epsilons for vizualization
epsilons = []

ddqn_agent = ddqn.Agent(learning_rate = 0.00025, input_dim = (8,), n_actions = 2, \
                        mem_size = 3000000, batch_size = 36, name = "ddqn", \
                            checkpoint_dir = "model_checkpoint")

#ddqn_agent.load_model() #loading a trained model

for n in range(1):    
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
        ddqn_agent.save_model()
        utility.save_object(scores, "scores")
        utility.save_object(epsilons, "epsilons")
        print(f"current epsilon: {ddqn_agent.epsilon}")

env.close()
utility.plot_learning_curve(scores, epsilons, filename = "iteration_2", path="plotting", mean_over=100)


