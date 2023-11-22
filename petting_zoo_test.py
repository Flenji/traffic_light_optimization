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

env = sumo_rl.parallel_env(net_file='Networks/basic_2lane.net.xml',
                  route_file='Networks/basic_2lane.rou.xml',
                  render_mode="human",
                  reward_fn=reward_fncs._combined_reward1,
                  observation_class=observation_spaces.ObservationFunction1,
                  use_gui=True,
                  num_seconds=1500)
# environment = AECEnv(env)
# environment.render_mode = "human"   
# env.env_params.additional_params.render_mode = "human"


scores = []
epsilons = []

ddqn_agent = ddqn.Agent(learning_rate = 0.00025, input_dim = (6,), n_actions = 2, \
                        mem_size = 3000000, eps_dec = 1e-6, batch_size = 36, name = "ddqn", \
                            checkpoint_dir = "model_checkpoint")

for n in range(10):    
    observations = env.reset()[0]
    print(f"Generation: {n}")
    while env.agents:
        actions = {agent: ddqn_agent.get_action(observations[agent]) for agent in env.agents}  # this is where you would insert your policy
        
        
        
        #actions = {ddqn_agent.get_action(obs)}
        #obs_, rewards, terminations, truncations, infos = 
        
        observations_, rewards, terminations, truncations, infos = env.step(actions)
        
        for agent in env.agents:
            obs = observations[agent]
            action = actions[agent]
            obs_, reward, termination, truncation, info = observations_[agent],\
                rewards[agent], terminations[agent], truncations[agent], infos[agent]
                
            done = termination or truncation
            
            
            ddqn_agent.learn(obs, action, reward, obs_, done)
            
        observations = observations_
        scores.extend(rewards.values())
        epsilons.append(ddqn_agent.epsilon)
    
    if n % 10 == 0:
        ddqn_agent.save_model()
        utility.save_object(scores, "scores")
        utility.save_object(epsilons, "epsilons")
        print(f"current epsilon: {ddqn_agent.epsilon}")


utility.plot_learning_curve(scores, epsilons, filename = "test", path="plotting")


