# Drone-Hover-RL
A hover system for Microsoft AirSim using reinforcement learning and Q-learning

Unmanned aerial vehicle (UAV) controller design is an important
building block in creating an autonomous UAV, and
it has been the subject of much research in recent years.
In this work, we present an approach to this challenge using
reinforcement learning (RL) combined with a neural network
to teach a simulated quadrotor how to hover. Using two
techniques, Q-learning and SARSA, we managed to build an
agent that can intelligently adjust its pose to hover in a certain
spot. Our results showed that combining RL with a neural
network and a suitable reward function can achieve our goal.

# How To Run
AirSim (in particular AirSim Neighborhood) is required to train the RL agent. We also used the Python client for AirSim, which is downloadable via the AirSim Github. Before running rl-agent.py, make sure AirSimClient.py is in the same directory and AirSim Neighborhood is running. 

When training, the weights for the hidden and output layers of the neural net will be saved as weights_hidden.txt and weights_output.txt, which can then be loaded back into the system by setting the readWeights boolean to True in rl-agent.py.
