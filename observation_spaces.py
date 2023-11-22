# -*- coding: utf-8 -*-
"""

"""

from sumo_rl.environment.observations import ObservationFunction
import numpy as np
from gymnasium import spaces
from sumo_rl.environment.traffic_signal import TrafficSignal

class ObservationFunction1(ObservationFunction):
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
    
class ObservationFunction2(ObservationFunction):
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