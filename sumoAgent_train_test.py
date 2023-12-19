# -*- coding: utf-8 -*-
"""
This module is used to train and test an agent or multi agent model.

Authors: AAU CS (IT) 07 - 03
"""

import sumo_rl
from pettingzoo import AECEnv
import matplotlib.pyplot as plt
import numpy as np
import ddqn
import utility
import os
import observation_spaces 
import observation_spaces 
import time
import reward_fncs

start_time = time.time()

net_file = 'Networks/single_agent_networks/1w/1w.net.xml'
test_suffix='_low'
route_file='Networks/single_agent_networks/1w/1w'+test_suffix+'.rou.xml'
observation_class = observation_spaces.ObservationFunction1
reward_fn = reward_fncs._combined_reward1

#set parameters for using sumolib in ComplexObservationFunction
#custom_observation.ComplexObservationFunction.net_file = net_file 
#custom_observation.ComplexObservationFunction.radius = 1
#custom_observation.ComplexObservationFunction.mode = "lane"

### SETTING HYPERPARAMETERS
learning_rate = 0.0025
mem_size = 3000000
eps_dec = 1e-5
batch_size = 36
gamma = 0.9
eps_min = 0.1
replace = 1000
checkpoint_dir = "model_checkpoint"
checkpoint_dir = utility.createPath("model_checkpoint", "multi_agent")

#Load or Save model?
SAVE = False
LOAD = True

env = sumo_rl.parallel_env(net_file=net_file,
                  route_file=route_file,
                  use_gui=False,
                  num_seconds=3600,
                  observation_class = observation_class,
                  reward_fn = reward_fn,
                  )

agent_suffix = "_1w_obs1_rew1"

### Setting the DDQN Agent for every possible agent
agents = dict.fromkeys(env.possible_agents)
scores = dict.fromkeys(env.possible_agents) # for plotting the learning curve
epsilons = []


for agent in agents.keys():
    
    scores[agent] = []
    
    input_shape = env.observation_space(agent).shape #gets number of input features for each agent
    n_actions = env.action_space(agent).n #number of possible actions
    name = agent + "_ddqn" + agent_suffix
    agents[agent] = ddqn.Agent(learning_rate=learning_rate, input_dim= input_shape, n_actions=n_actions,\
                               mem_size=mem_size, eps_dec=eps_dec, eps_min = eps_min, gamma = gamma,\
                                   batch_size= batch_size, name = name, checkpoint_dir= checkpoint_dir,\
                                       replace = replace)
    if LOAD:
        agents[agent].load_model()
        
        scores = utility.load_object("scores"+agent_suffix, "results")
        epsilons = utility.load_object("epsilons"+ agent_suffix, "results")

print(f"Agents in this simulation: {[a for a in agents.keys()]}")

num_simulations = 175

def train(num_simulations):
    """
    Trains the agents for minimum min_learning_steps. If the one learning episode ends (the simulations ends)
    and the ammount of learning steps taken is >= min_learning_steps the training is done.
    """
    learning_steps = 0
    for n in range(num_simulations):
        observations = env.reset()[0]
        print(f"Generation: {n}")
        while env.agents: #contains agents as long simulation is running

            actions =  {agent: agents[agent].get_action(observations[agent]) for agent in env.agents}
            
            observations_, rewards, terminations, truncations, infos = env.step(actions)
                
            for agent in env.agents:
                obs = observations[agent] #current observation of agent
                action = actions[agent] 
                obs_, reward, termination, truncation, info = observations_[agent],\
                    rewards[agent], terminations[agent], truncations[agent], infos[agent]
                    
                done = termination or truncation #this is not necessary in this environment because there is no "end" of traffic
                
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
    
    utility.plot_learning_curves(scores, epsilons, 3, 3, filename = "model_1200"+agent_suffix, path="results", mean_over=1200)


def test(random = False, metrics = False, use_gui = True, test_name = ""):
    """
    Function test the agents. If random = True, agents chose just random actions.
    """
    
    if metrics:
        additional_sumo_cmd = "--additional-files additional.xml"
    else:
        additional_sumo_cmd = ""
    
    env = sumo_rl.parallel_env(net_file=net_file,
                      route_file=route_file,
                      use_gui=use_gui,
                      num_seconds=simulation_seconds,
                      observation_class = observation_class,#ComplexObservationFunction,
                      reward_fn = "average-speed",#reward_fncs.multi_agent_reward3, # "average-speed",
                      additional_sumo_cmd = additional_sumo_cmd,#,"--edgedata-output metrics.xml",
                      sumo_seed = 0,
                      )
    
    
    observations = env.reset()[0]
    
    while env.agents:
        if random:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        else:
            actions =  {agent: agents[agent].get_test_action(observations[agent]) for agent in env.agents}
        
        observations_, rewards, terminations, truncations, infos = env.step(actions)
        observations = observations_ #setting new observation as current observation
        
    if metrics:
        file_name_old = utility.createPath("metrics","metrics.xml")
        file_name_new = utility.createPath("metrics","metrics"+agent_suffix+"_"+test_name+".xml")
        os.rename(file_name_old,file_name_new)

#train(min_learning_steps)
env.close()

end_time = time.time()

print(f"Runtime {utility.get_time_formatted(end_time-start_time)}")

#test(metrics=True,use_gui= False, test_name="all")

