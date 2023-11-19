# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:44:36 2023

@author: hanne
"""

from sumo_rl.environment.observations import ObservationFunction
import numpy as np
from gymnasium import spaces
from sumo_rl.environment.traffic_signal import TrafficSignal

class CustomObservationFunction(ObservationFunction):
    """Custom observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize  observation function."""
        super().__init__(ts)
        
    def __call__(self) -> np.ndarray:
        """Return the observation."""
        density = self.ts.get_lanes_density()
        observation = np.array(density, dtype=np.float32)
        return observation
    
    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(len(self.ts.lanes), dtype=np.float32),
            high=np.ones(len(self.ts.lanes), dtype=np.float32),
        )