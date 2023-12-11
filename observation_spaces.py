"""
    This module contains the definition of several observation spaces designed for SUMO Reinforcement Learning.

    Authors: AAU CS (IT) 07 - 03
"""

from sumo_rl.environment.observations import ObservationFunction
import numpy as np
from gymnasium import spaces
from sumo_rl.environment.traffic_signal import TrafficSignal
import aux_functions
import sumolib.net as net

#### EDGES OBSERVATION FUNCTIONS

class ObservationFunction1(ObservationFunction):
    """The agent receives the congestion of each of the edges aproaching the intersection.
       DIM: 9 (if there are 4 green phases)  (for 4 incoming edges)"""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)
        aux_functions._additional_tls_info(self.ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = aux_functions.get_incoming_edges_density(self.ts, self.ts.incoming_edges)
        observation = np.array(phase_id + min_green + density, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + len(self.ts.incoming_edges), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + len(self.ts.incoming_edges), dtype=np.float32),
        )
    

class ObservationFunction2(ObservationFunction):
    """The agent receives the congestion of each of the edges aproaching the intersection.
       It also receives the number of queued vehicles of each of the edges aproaching the intersection. 
       DIM: 13 (for 4 incoming edges)
       
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
        density = aux_functions.get_incoming_edges_density(self.ts, self.ts.incoming_edges)
        queued = aux_functions.get_edges_queue(self.ts, self.ts.incoming_edges) # (self.ts.get_lanes_queue() to get the sum of queued vehicles)
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
    """The agent receives the congestion of each of the edges aproaching the intersection.
       It also receives the number of queued vehicles of each of the edges aproaching the intersection.
       In addition, the number of lanes of each of the edges aproaching the intersection is given.
       DIM: 17 (for 4 incoming edges)
       
       OBS: By adding the number of lanes of each edge we give the agent the ability to perceive how the network is. Therefore, it can know if there are
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
        density = aux_functions.get_incoming_edges_density(self.ts, self.ts.incoming_edges)
        queued = aux_functions.get_edges_queue(self.ts, self.ts.incoming_edges) # (self.ts.get_lanes_queue() to get the sum of queued vehicles)
        lanes = aux_functions.get_incoming_num_lanes_per_edge(self.ts, self.ts.incoming_edges)
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
    

#### LANES OBSERVATION FUNCTIONS

class ObservationFunction2_lanes(ObservationFunction):
    """The agent receives the congestion of each of the edges aproaching the intersection.
       It also receives the number of queued vehicles of each of the edges aproaching the intersection. 
       DIM: 21 (for 8 incoming lanes)
       
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
        density = aux_functions.get_incoming_lanes_density(self.ts, self.ts.in_lanes)
        queued = aux_functions.get_lanes_queue(self.ts, self.ts.in_lanes) # (self.ts.get_lanes_queue() to get the sum of queued vehicles)
        observation = np.array(phase_id + min_green + density + queued, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.in_lanes), dtype=np.float32),
            high=np.concatenate([
                np.ones(self.ts.num_green_phases + 1 + len(self.ts.in_lanes), dtype=np.float32),
                np.full(len(self.ts.in_lanes), np.inf, dtype=np.float32)  # Set the upper bound to positive infinity for queue
            ])
        )
    

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

class ComplexObservationFunction(ObservationFunction):
    """
    Custom observation function for traffic signals. The observation function
    takes lanes and traffic lights in a given radius n into account.
    """
    
    net_file = ""
    radius = 0
    
    mode = "lane"
    
    ts = None
    compObject = None 
    
    def __init__(self, ts: TrafficSignal):
        """Initialize  observation function."""
        super().__init__(ts)

        self.sumo = self.ts.sumo
        lanes = list(
            dict.fromkeys(self.sumo.trafficlight.getControlledLanes(ts.id))
        )
        
        self.direct_controlled_lanes = lanes
        self.net = net.readNet(ComplexObservationFunction.net_file)
        self.mode = ComplexObservationFunction.mode
        
        radius = ComplexObservationFunction.radius
        self.controlled_lanes = self.getControlledLanesRadius(lanes,radius)
        
        #self.outgoing_lanes = self.getOutgoingLanesRadius(lanes,radius)
        
        #self.controlled_lanes = list(dict.fromkeys(self.controlled_lanes+self.outgoing_lanes))
        
        self.connected_traffic_lights = [tl for tl in self.connectedTrafficLights() if tl != ts.id]
        
        self.tl_green_phases = self.getConnectedTrafficLightPhases()
        
        self.count = 0
        
        self.lanes_lenght = {lane: self.sumo.lane.getLength(lane) for lane in self.controlled_lanes}#+self.outgoing_lanes}
        
        #self.controlled_lanes = list(dict.fromkeys(self.controlled_lanes + self.outgoing_lanes))
        
        #Setting Edges for Reward Function
        self.direct_controlled_edges = self.lanesToEdges(self.direct_controlled_lanes)
        self.controlled_edges = self.lanesToEdges(self.controlled_lanes)
        
        self.edges_length = self.getEdgesLength(self.controlled_edges)
        
        self.ts.direct_controlled_edges = self.direct_controlled_edges
        self.ts.controlled_edges = self.controlled_edges
        
        self.ts.direct_controlled_lanes = lanes
        self.ts.controlled_lanes = self.controlled_lanes
        
        ComplexObservationFunction.ts = self.ts
        ComplexObservationFunction.compObject = self
        

    def lanesToEdges(self,lanes):
        """
        Gets the corresponding edges to the lanes.
        """
        edge_ids = []

        for lane in lanes:
            lane_object = self.net.getLane(lane)
            edge_object = lane_object.getEdge()
            edge_ids.append(edge_object.getID())

        edge_ids = sorted(dict.fromkeys(edge_ids))
        return edge_ids
    
    def getEdgesLength(self, edges):
        """
        Gets the length of every edge in edges.
        Returns a dictionary.
        """
        edges_length = {}
        for edge in edges:
            edge_object = self.net.getEdge(edge)
            edges_length[edge] = edge_object.getLength()
        return edges_length
            
                
    def getControlledLanesRadius(self,controlled_lanes,radius):
        """
        This function gets a list of lanes and expands the list of lanes with
        the incoming next n incoming lanes where n = radius. It returns the expanded list.
        """
        con_lanes_list = [controlled_lanes] 
        for i in range(radius):
            next_lanes = [] 
            for lane in con_lanes_list[-1]: #for lane in the list of last calculated lanes
                lane_object = self.net.getLane(lane)
                next_lanes.extend( [l.getID() for l in lane_object.getIncoming()])
            con_lanes_list.append(next_lanes)
            
        controlled_lanes = sorted({l for sublist in con_lanes_list for l in sublist}) #to remove duplicates
            
        return controlled_lanes
    
    def getOutgoingLanesRadius(self,lanes, radius):
        """
        This function gets a list of lanes and expands the list of lanes with
        the outgoing next n incoming lanes where n = radius. It returns the expanded list.
        """
        out_going = []
        for lane in lanes: #get the first outgoing lanes corresponding to the standard controlled lanes
            lane_object = self.net.getLane(lane)
            outgoingLanesObj = lane_object.getOutgoingLanes()
            
            outgoingLanes = [outgoing.getID() for outgoing in outgoingLanesObj]
            
            out_going.extend(outgoingLanes)

        out_going = list(dict.fromkeys(out_going))
        
        con_lanes_list = [out_going] 
        for i in range(radius): #expanding that list with the outgoing lanes 
            next_lanes = [] 
            for lane in con_lanes_list[-1]: #for lane in the list of last calculated lanes
                lane_object = self.net.getLane(lane)
                next_lanes.extend( [l.getID() for l in lane_object.getOutgoingLanes()])
            con_lanes_list.append(next_lanes)
            
        controlled_lanes = sorted({l for sublist in con_lanes_list for l in sublist})
        
        return controlled_lanes
    
    def connectedTrafficLights(self):
        """
        Checks for every lane in self.controlled_lanes if that lane is controlled by a traffic light
        if yes, that traffic light will be added to the list of connected_traffic_lights.
        """
        connected_traffic_lights = []
        traffic_lights = [tl.getID() for tl in self.net.getTrafficLights()]
        for tl in traffic_lights:      
            tl_controlled_lanes = self.sumo.trafficlight.getControlledLanes(tl)
            for lane in tl_controlled_lanes:
                if lane in self.controlled_lanes:
                    #print(tl)
                    #print("traffic_light connected")
                    connected_traffic_lights.append(tl)
                    break
        
        return connected_traffic_lights
        
    def getConnectedTrafficLightPhases(self):
        """
        Returns a dictionary that contains every possible green phase configuration 
        for each traffic light.
        """
        tl_green_phases = {}
        
        for tl in self.connected_traffic_lights:
            phases = self.sumo.trafficlight.getAllProgramLogics(tl)[0].phases
            green_phases = []
            num_green_phases = 0
            for phase in phases:
                state = phase.state
                if "y" not in state and (state.count("r") + state.count("s") != len(state)):
                    num_green_phases += 1
                    green_phases.append(state)
            tl_green_phases[tl] = green_phases
        return tl_green_phases
    
    def getPhaseIDs(self):
        """
        Returns the current phaseIDs in the simulation. That means it return which traffic phase
        is green at the moment in an one-hot-encoded shape.
        """
        phase_IDs = []
        for tl in self.connected_traffic_lights:
            current_state = self.sumo.trafficlight.getRedYellowGreenState(tl)
            phase_id = [1  if current_state == state else 0 for state in self.tl_green_phases[tl]] #one-hot encoding
            phase_IDs.append(phase_id)
        phase_IDs = [i for sublist in phase_IDs for i in sublist] # flatten list for observation return
        return phase_IDs
    
    
    def get_lanes_density(self,lanes):
        """Returns the density [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        """
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_lenght[lane] / (self.ts.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in lanes
        ]
        return [min(1, density) for density in lanes_density]
    
    def get_edges_density(self,edges):
        edges_density = [
            self.sumo.edge.getLastStepVehicleNumber(edge)
            / ((self.edges_length[edge]) /(self.ts.MIN_GAP + self.sumo.edge.getLastStepLength(edge)))
            for edge in edges
            ]

        return [min (1, density) for density in edges_density]

        
    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        

        len_phase_id_encodings = \
            sum([len(green_phase) for tl, green_phase in self.tl_green_phases.items()])
        
        
        if self.mode == "lane":
            n_objects = len(self.controlled_lanes)
        elif self.mode == "edge":
            n_objects = len(self.controlled_edges)
            
        return spaces.Box(
            low=np.zeros(len_phase_id_encodings+n_objects, dtype=np.float32),
            high=np.ones(len_phase_id_encodings+n_objects, dtype=np.float32),
        )
    
    def __call__(self) -> np.ndarray:
        """Return the observation."""
        
        phaseIDs = self.getPhaseIDs()
        
        if(self.mode =="lane"):
            density = self.get_lanes_density(self.controlled_lanes)
        elif(self.mode == "edge"):
            density = self.get_edges_density(self.controlled_edges)
        #density_outgoing = self.get_lanes_density(self.outgoing_lanes)
        
        observation = np.array(phaseIDs+density, dtype=np.float32)
        return observation
    