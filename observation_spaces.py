"""
    This module contains the definition of several observation spaces designed for SUMO Reinforcement Learning.

    Authors: AAU CS (IT) 07 - 03
"""

from sumo_rl.environment.observations import ObservationFunction
import numpy as np
from gymnasium import spaces
from sumo_rl.environment.traffic_signal import TrafficSignal
import aux_functions

class ObservationFunction1(ObservationFunction):
    """The agent receives the congestion of each of the 4 edges aproaching the intersection.
       DIM: 9 (if there are 4 green phases)"""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)
        aux_functions._additional_tls_info(self.ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = aux_functions.get_incoming_edges_density(self.ts)
        observation = np.array(phase_id + min_green + density, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + len(self.ts.incoming_edges), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + len(self.ts.incoming_edges), dtype=np.float32),
        )
    

class ObservationFunction2(ObservationFunction):
    """The agent receives the congestion of each of the 4 edges aproaching the intersection.
       It also receives the number of queued vehicles of each of the 4 edges aproaching the intersection. 
       DIM: 13
       
       OBS: By not getting the queue in a percentage, and combining with the density in the observation space, the agent will get a notion
            of the measures of the edges without having to increase the obseravtion space with the number of lanes per edge and their length.
            (IF WE SEE THAT THE QUEUE DOES NOT HELP REFLECTING THAT, WE CAN ADD THE NUMBER OF VEHICLES IN THE EDGE)
    """

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)
        aux_functions._additional_tls_info(self.ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = aux_functions.get_incoming_edges_density(self.ts)
        queued = aux_functions.get_edges_queue(self.ts) # (self.ts.get_lanes_queue() to get the sum of queued vehicles)
        observation = np.array(phase_id + min_green + density + queued, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.incoming_edges), dtype=np.float32),
            high=np.concatenate([
                np.ones(self.ts.num_green_phases + 1 + len(self.ts.incoming_edges), dtype=np.float32),
                np.full(len(self.ts.incoming_edges), np.inf, dtype=np.float32)  # Set the upper bound to positive infinity for queue
            ])
        )
    
class ObservationFunction3(ObservationFunction):
    """The agent receives the congestion of each of the 4 edges aproaching the intersection.
       It also receives the number of queued vehicles of each of the 4 edges aproaching the intersection.
       In addition, the number of lanes of each of the 4 edges aproaching the intersection is given.
       DIM: 17
       
       OBS: By addin the number of lanes of each edge we give the agent the ability to perceive how the network is. Therefore, it can know if there are
       more important edges and is able to have a sense of how many cars are there in each edge with the help of the other inputs.
    """

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)
        aux_functions._additional_tls_info(self.ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = aux_functions.get_incoming_edges_density(self.ts)
        queued = aux_functions.get_edges_queue(self.ts) # (self.ts.get_lanes_queue() to get the sum of queued vehicles)
        lanes = aux_functions.get_incoming_num_lanes_per_edge(self.ts)
        observation = np.array(phase_id + min_green + density + queued + lanes, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.concatenate([
                np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.incoming_edges), dtype=np.float32),
                np.ones(len(self.ts.incoming_edges), dtype=np.float32)
            ]),
            high=np.concatenate([
                np.ones(self.ts.num_green_phases + 1 + len(self.ts.incoming_edges), dtype=np.float32),
                np.full(2 * len(self.ts.incoming_edges), np.inf, dtype=np.float32)  # Set the upper bound to positive infinity for queue
            ])
        )