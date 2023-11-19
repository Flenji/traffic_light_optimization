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
from .traffic_signal import TrafficSignal

### OBSERVATION CLASSES

class ObservationFunction1(sumo_rl.environment.observations.ObservationFunction):
    """The agent receives the congestion of each of the 4 edges aproaching the intersection.
       DIM: 6"""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_incoming_edges_density()
        observation = np.array(phase_id + min_green + density, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )
    
class ObservationFunction2(sumo_rl.environment.observations.ObservationFunction):
    """The agent receives the congestion of each of the 4 edges aproaching the intersection.
       It also receives the number of queued vehicles of each of the 4 edges aproaching the intersection. 
       DIM: 11
       
       OBS: By not getting the queue in a percentage, and combining with the density in the observation space, the agent will get a notion
            of the measures of the edges without having to increase the obseravtion space with the number of lanes per edge and their length.
            (IF WE SEE THAT THE QUEUE DOES NOT HELP REFLECTING THAT, WE CAN ADD THE NUMBER OF VEHICLES IN THE EDGE)
    """

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_incoming_edges_density()
        queued = self.ts.get_edges_queue() # (self.ts.get_lanes_queue() to get the sum of queued vehicles)
        observation = np.array(phase_id + min_green + density + queued, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )


### ENVIRONMENT

env = sumo_rl.parallel_env(net_file='basicnet_2lane.net.xml',
                  route_file='simple1.rou.xml',
                  reward_fn="combined1",
                  observation_class=ObservationFunction2,
                  use_gui=False,
                  num_seconds=3000)
# environment = AECEnv(env)
# environment.render_mode = "human"   
# env.env_params.additional_params.render_mode = "human"


scores = []
epsilons = []

ddqn_agent = ddqn.Agent(learning_rate = 0.0025, input_dim = (19,), n_actions = 2, \
                        mem_size = 100000, batch_size = 8, name = "ddqn", \
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


