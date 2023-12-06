# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:07:50 2023

@author: hanne
"""

import sumo_rl
from pettingzoo import AECEnv
import matplotlib.pyplot as plt
import numpy as np
import ddqn
import utility
import os
import custom_observation 
import time

start_time = time.time()

net_file = 'fumos/V/V.net.xml'
#set parameters for using sumolib in ComplexObservationFunction
custom_observation.ComplexObservationFunction.net_file = net_file 
custom_observation.ComplexObservationFunction.radius = 1
custom_observation.ComplexObservationFunction.mode = "lane"

env = sumo_rl.parallel_env(net_file=net_file,
                  route_file='fumos/V/V.rou.xml',
                  use_gui=True,
                  num_seconds=3600,
                  observation_class = custom_observation.ComplexObservationFunction,#ComplexObservationFunction,
                  reward_fn = "average-speed",
                  )


### SETTING HYPERPARAMETERS
learning_rate = 0.0025
mem_size = 1000000
eps_dec = 5e-6*3
batch_size = 36
gamma = 0.99
eps_min = 0.1
replace = 1000
checkpoint_dir = utility.createPath("model_checkpoint", "sixth_iteration")
SAVE = False#True
LOAD = False


### Setting the DDQN Agent for every possible agent
agents = dict.fromkeys(env.possible_agents)
scores = dict.fromkeys(env.possible_agents) # for plotting the learning curve
epsilons = []

agent_suffix = "V_complex"
for agent in agents.keys():
    
    scores[agent] = []
    
    input_shape = env.observation_space(agent).shape
    n_actions = env.action_space(agent).n
    name = agent + "_ddqn" + "_complexObs" + agent_suffix
    agents[agent] = ddqn.Agent(learning_rate=learning_rate, input_dim= input_shape, n_actions=n_actions,\
                               mem_size=mem_size, eps_dec=eps_dec, eps_min = eps_min, gamma = gamma,\
                                   batch_size= batch_size, name = name, checkpoint_dir= checkpoint_dir,\
                                       replace = replace)
    if LOAD:
        agents[agent].load_model()

print(f"Agents in this simulation: {[a for a in agents.keys()]}")

min_learning_steps = 220000/3

def train(min_learning_steps):
    """
    Trains the agents for minimum min_learning_steps. If the one learning episode ends (the simulations ends)
    and the ammount of learning steps taken is >= min_learning_steps the training is done.
    """
    learning_steps = 0
    n = 0
    while(learning_steps <= min_learning_steps):
        observations = env.reset()[0]
        print(f"Generation: {n}")
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
                scores[agent].append(reward)
                
            epsilons.append(agents[agent].epsilon)    
            observations = observations_ #setting new observation as current observation
            
            learning_steps += 1
            
        if n % 10 == 0:
            if SAVE:
                for k,v in agents.items():
                    v.save_model()
                utility.save_object(scores, "scores"+agent_suffix, "results")
                utility.save_object(epsilons, "epsilons"+ agent_suffix, "results")
            print(f"current epsilon: {epsilons[-1]}")
            print(f"learning steps taken: {learning_steps}")
        n += 1

def test(random = False):
    """
    Function test the agents. If random = True, agents chose just random actions.
    """
    observations = env.reset()[0]
    
    while env.agents:
        if random:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        else:
            actions =  {agent: agents[agent].get_test_action(observations[agent]) for agent in env.agents}
        
        observations_, rewards, terminations, truncations, infos = env.step(actions)
        observations = observations_ #setting new observation as current observation


#train(min_learning_steps)
test(False)
env.close()

end_time = time.time()

print(f"Runtime {utility.get_time_formatted(end_time-start_time)}")

#utility.plot_learning_curves(scores, epsilons, 3, 3, filename = "test_1000"+agent_suffix, path="results", mean_over=1000)
