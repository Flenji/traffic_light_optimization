# Dynamic traffic control with Double Deep Q-Learning 
This is the repository containing the scripts and files related to the research portraied by the paper "Dynamic traffic control with Double Deep Q-Learning".  

### Prerequisites

Before running scripts of this project, make sure to have the reqiured packages and programs installed. 

- Simulation of Urban MObility (SUMO), can be found https://eclipse.dev/sumo/

#### Python Packages
- pytorch
- gymnasium
- numpy
- matplotlib
- sumo_rl
- pettingZoo


## Training an Agent

The scripts sumoAgent_rand_train_test.py and sumoAgent_train_test.py are used to train the agents. As the name implies, sumoAgent_rand_train uses random traffic flows while sumoAgent_train_test uses the flows predefined in the network files.

To train an agent, different parametets can be set. The follown example shows different paramters with example values:

- net_file = 'Networks/VI/VI.net.xml' -- sumo network file 
- route_file='Networks/VI/VI_all_fix.rou.xml' -- sumo route file
- observation_class = observation_spaces.CustomObservationFunction -- sets which observation class should be used
- simulation_seconds = 6000 -- lenth of each simulation
- reward_fn = 'average-speed' -- reward function that is used
- SAVE = True -- If the model parameters should be saved
- LOAD = True -- If an already save model should be loaded
- agent_suffix = "_sObs_sRew" -- defines the name of the model (important for loading and saving)

There are also many hyperparameters that can be set in the script.

To train an againt, the function train(min_learning_steps) is used. min_learning_steps is the minimum amount of learning steps the agents should be trained. After that amount is reached the agents finishes the current training epsisode and stops training.

To test the agent, the function test(random = False, metrics = False, use_gui = True, test_name = "") is used. random specifies if the agent should act completely random, metrics specifies if an metrics about the test run should be saved as a xml-file. use_gui if the grpahical interface is used.

## Different Observation Spaces
Different observation spaces such as the complex or simple observation space can be found in the observation_spaces.py file.

## Reward Functions

Different reward functions are defined in reward_fncs.py.

## Evaluate the agents
In the metrics folder, the file comparison.py is used to evaluate the metric-xml files. 
As an example, running the comparison script with the argument foldername = "high_flow" specifies that only the xml-files in the high_flow folder should be evaluated.

