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

from lxml import etree as ET
import random


def generate_random_partition(total_sum, num_elements):
    """
    """
    # Generate num_elements random numbers between 0 and total_sum
    random_numbers = random.sample(range(total_sum), num_elements)

    # Ensure that the total sum of numbers adds up to 100
    while sum(random_numbers) > total_sum:
        index_to_adjust = random.randint(0, num_elements - 1)
        if random_numbers[index_to_adjust] > 3:
            random_numbers[index_to_adjust] -= 1
    while sum(random_numbers) < total_sum:
        index_to_adjust = random.randint(0, num_elements - 1)
        random_numbers[index_to_adjust] += 1
        
    return random_numbers


def define_new_flows(route_file, number_routes, total_vehsPerHour, total_routeProbs):
    """
    """
    # Read XML file
    tree = ET.parse(route_file)
    root = tree.getroot()

    # Generate random values
    vehsPerHour = round(float(random.randint(1800, 3600)), 2)
    total_vehsPerHour += vehsPerHour
    route_probabilities = generate_random_partition(100, number_routes)

    # Modify attributes in the route file with the random values
    root.find('.//flow[@id="f_0"]').set('vehsPerHour', str(vehsPerHour))
    for i in range(len(route_probabilities)):
        route = './/route[@id="r_' + str(i) + '"]'
        root.find(route).set('probability', str(route_probabilities[i]))
        total_routeProbs[i] += route_probabilities[i]

    # Write updated XML back to the file
    tree.write(route_file)

    return total_vehsPerHour, total_routeProbs


scores = [] #utility.load_object("scores") #keeping track of scores and epsilons for vizualization
epsilons = [] #utility.load_object("epsilons")

total_vehsPerHour = 0
total_routeProbs = [0 for i in range(10)]

ddqn_agent = ddqn.Agent(learning_rate = 0.0001, input_dim = (21,), n_actions = 4, \
                        mem_size = 3000000, eps_dec = 1e-6, batch_size = 36, name = "ddqn9", \
                            checkpoint_dir = "model_checkpoint", gamma=0.9)

for n in range(500):    
    
    total_vehsPerHour, total_routeProbs = define_new_flows('Networks/second_random.rou.xml', 10, total_vehsPerHour, total_routeProbs)

    env = sumo_rl.parallel_env(net_file='Networks/second.net.xml',
                  route_file='Networks/second_random.rou.xml',
                  reward_fn=reward_fncs._combined_reward4,
                  observation_class=observation_spaces.ObservationFunction2_lanes,
                  use_gui=False,
                  num_seconds=10800)
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
        ddqn_agent.save_model("model9")
        utility.save_object(scores, "scores9")
        utility.save_object(epsilons, "epsilons9")
        print(f"current epsilon: {ddqn_agent.epsilon}")

    env.close()

utility.plot_learning_curve(scores, epsilons, filename = "model9", path="plotting", mean_over=1000)