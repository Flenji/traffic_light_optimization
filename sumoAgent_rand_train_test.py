# -*- coding: utf-8 -*-
"""
This module is used to train a single agent over a specific network design with a variety of random flows and test it in a specific scenario.

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
import time
import reward_fncs

import argparse

from lxml import etree as ET
import random


def generate_random_partition(total_sum, num_elements):
    """
    """
    # Generate num_elements random numbers between 0 and total_sum
    #random_numbers = random.sample(range(total_sum), num_elements)
    random_numbers = random.choices(range(3, 31), k=num_elements)

    # Ensure that the total sum of numbers adds up to 100
    while sum(random_numbers) > total_sum:
        index_to_adjust = random.randint(0, num_elements - 1)
        if random_numbers[index_to_adjust] > 3:
            random_numbers[index_to_adjust] -= 1
    while sum(random_numbers) < total_sum:
        index_to_adjust = random.randint(0, num_elements - 1)
        random_numbers[index_to_adjust] += 1
        
    return random_numbers


def define_new_flows(route_file, number_routes):
    """
    """
    # Read XML file
    tree = ET.parse(route_file)
    root = tree.getroot()

    # Generate random values
    vehsPerHour = round(float(random.randint(400, 2300)), 2)
    route_probabilities = generate_random_partition(100, number_routes)

    # Modify attributes in the route file with the random values
    root.find('.//flow[@id="random"]').set('vehsPerHour', str(vehsPerHour))
    for i in range(len(route_probabilities)):
        route = './/route[@id="r_' + str(i) + '"]'
        root.find(route).set('probability', str(route_probabilities[i]))

    # Write updated XML back to the file
    tree.write(route_file)


start_time = time.time()

############## Batch testing

parser = argparse.ArgumentParser()
parser.add_argument('--test_suffix', type=str)
parser.add_argument('--rew_suffix', type=str)
parser.add_argument('--seed', type=str)
args = parser.parse_args()

net_file = 'Networks/single_agent_networks/1w/1w.net.xml'
train_route_file = 'Networks/single_agent_networks/1w/1w_random.rou.xml'
test_suffix=args.test_suffix
rew_suffix=args.rew_suffix
seed=args.seed
test_route_file='Networks/single_agent_networks/1w/1w'+test_suffix+'.rou.xml'
observation_class = observation_spaces.ObservationFunction2
reward_function = getattr(reward_fncs, f'_combined_reward{rew_suffix}')
num_seconds = 3600

agent_suffix = "_reward"+rew_suffix+"_randtraining"

##############
"""
net_file = 'Networks/single_agent_networks/1w/1w.net.xml'
train_route_file = 'Networks/single_agent_networks/1w/1w_random.rou.xml'
test_suffix='_low'
test_route_file = 'Networks/single_agent_networks/1w/1w'+test_suffix+'.rou.xml'
observation_class = observation_spaces.ObservationFunction2
reward_function = reward_fncs._combined_reward3
num_seconds = 7200

agent_suffix = "_reward3_randtraining"
"""
### SETTING HYPERPARAMETERS
learning_rate = 0.0001
mem_size = 3000000
eps_dec = 1.5e-6
batch_size = 36
gamma = 0.9
eps_min = 0.1
replace = 1000
checkpoint_dir = "model_checkpoint"

#Load or Save model?
SAVE = False
LOAD = True

epsilons = []
scores = []

name = "ddqn" + agent_suffix
ddqn_agent = ddqn.Agent(learning_rate=learning_rate, input_dim= (21,), n_actions=4,\
                       mem_size=mem_size, eps_dec=eps_dec, eps_min = eps_min, gamma = gamma,\
                       batch_size= batch_size, name = name, checkpoint_dir= checkpoint_dir,\
                       replace = replace, deeper=True)

if LOAD:
        ddqn_agent.load_model()

num_simulations = 450

def train(num_simulations):
    """
    Trains the agents for minimum min_learning_steps. If the one learning episode ends (the simulations ends)
    and the ammount of learning steps taken is >= min_learning_steps the training is done.
    """
    for n in range(num_simulations):

        define_new_flows(train_route_file, 12)

        env = sumo_rl.parallel_env(net_file= net_file,
                route_file=train_route_file,
                reward_fn=reward_function,
                observation_class=observation_class,
                use_gui=False,
                num_seconds=num_seconds)
    
        observations = env.reset()[0]
        print(f"Generation: {n}")
        while env.agents: #contains agents as long simulation is running

            actions =  {agent: ddqn_agent.get_action(observations[agent]) for agent in env.agents}
            
            observations_, rewards, terminations, truncations, infos = env.step(actions)
                
            for agent in env.agents:
                obs = observations[agent] #current observation of agent
                action = actions[agent] 
                obs_, reward, termination, truncation, info = observations_[agent],\
                    rewards[agent], terminations[agent], truncations[agent], infos[agent]
                    
                done = termination or truncation #TODO: see if this is needed for SUMO
                
                
                ddqn_agent.learn(obs, action, reward, obs_, done)
                
            scores.append(reward)
            epsilons.append(ddqn_agent.epsilon)    
            observations = observations_ #setting new observation as current observation
            
            
        if n % 10 == 0:
            if SAVE:
                ddqn_agent.save_model()
                utility.save_object(scores, "scores"+agent_suffix, "results")
                utility.save_object(epsilons, "epsilons"+ agent_suffix, "results")
            print(f"current epsilon: {ddqn_agent.epsilon}")

        env.close()

    utility.plot_learning_curve(scores, epsilons, filename = "model_"+agent_suffix, path="results", mean_over=2400)


def test(random = False, metrics = False, use_gui = True):
    """
    Function test the agent. If random = True, agents choose just random actions.
    """
    
    if metrics:
        additional_sumo_cmd = "--additional-files additional.xml"
    else:
        additional_sumo_cmd = ""
    
    env = sumo_rl.parallel_env(net_file=net_file,
                      route_file=test_route_file,
                      use_gui=use_gui,
                      num_seconds=num_seconds,
                      observation_class = observation_class,
                      reward_fn = reward_function,
                      additional_sumo_cmd = additional_sumo_cmd,
                      sumo_seed = seed
                      )
    
    observations = env.reset()[0]
    
    while env.agents:
        if random:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        else:
            actions =  {agent: ddqn_agent.get_test_action(observations[agent]) for agent in env.agents}
        
        observations_, rewards, terminations, truncations, infos = env.step(actions)
        observations = observations_ #setting new observation as current observation
    
    env.close()
    
    if metrics:
        file_name_old = utility.createPath("metrics","metrics.xml")
        file_name_new = utility.createPath("metrics","metrics"+agent_suffix+test_suffix+"_"+seed+".xml")
        #file_name_new = utility.createPath("metrics","metrics"+agent_suffix+test_suffix+".xml")
        os.rename(file_name_old,file_name_new)

#train(num_simulations)

end_time = time.time()

print(f"Runtime {utility.get_time_formatted(end_time-start_time)}")

test(metrics=True,use_gui= False)
