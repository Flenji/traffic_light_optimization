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

env = sumo_rl.parallel_env(net_file='Networks/second.net.xml',
                  route_file='Networks/second.rou.xml',
                  reward_fn=reward_fncs._combined_reward3,
                  observation_class=observation_spaces.ObservationFunction2_lanes,
                  use_gui=True,
                  num_seconds=3600)
# environment = AECEnv(env)
# environment.render_mode = "human"   
#env.env_params.additional_params.render_mode = "human"

ddqn_agent = ddqn.Agent(learning_rate = 0.0001, input_dim = (21,), n_actions = 4, \
                        mem_size = 3000000, eps_dec = 1e-6, batch_size = 36, name = "ddqn8", \
                            checkpoint_dir = "model_checkpoint", gamma=0.9)

ddqn_agent.load_model("model8") #loading a trained model

# Reset the environment to get the initial observations
observations = env.reset()[0]

# Run the simulation with the trained agent
while env.agents:
    actions = {agent: ddqn_agent.get_action(observations[agent]) for agent in env.agents}

    observations_, rewards, terminations, truncations, infos = env.step(actions)

    # Update observations for the next step
    observations = observations_
        

env.close()
