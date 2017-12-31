import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from AirSimClient import *
import sys
import time
import random
import msvcrt

np.set_printoptions(threshold=np.nan)

# if true, use Q-learning. Else, use SARSA
qlearning = True
readWeights = True # read in saved weights to resume progress

# drone boundaries
goal_limit = 0.5
max_radius = 3

# Set learning parameters
y = 0.1 # discount rate
e = 0.2 # epsilon
target_z = -2 # target height in NED coordinate system
num_episodes = 50000
episode_length = 100 # number of actions per episode

# ANN parameters
step_size = 2.5 # action space in increments of degrees
num_increments = 5
translate_scale = -5
num_outputs = num_increments**2
num_inputs = 6
learning_rate = 0.001
num_hidden = 10

def reward(state):
    # if did_reach_goal(state):
    if is_in_bounds_3d(state[:3], goal_limit):
        return 10
    else:
        return -50
def did_reach_goal(state):
    return is_in_bounds_3d(state[:3], goal_limit)
# takes an index of the action space (max Q) and converts to the action values a = (roll, pitch)
def get_action(index):
    return (normalize_deg((index // num_increments)*step_size + translate_scale), 
        normalize_deg((index % num_increments)*step_size + translate_scale))
def normalize_deg(x):
    return x/90
def scale_pos(s):
    pos_scaler = 5
    return [[ s[0]/pos_scaler, s[1]/pos_scaler, s[2]/pos_scaler, s[3], s[4], s[5] ]]
def distance(x, y):
    ref_x = 0
    ref_y = 0
    return np.sqrt((ref_x - x)**2 + (ref_y - y)**2)
def distance_3d(pos):
    x1 = 0; y1 = 0; z1 = target_z
    return np.sqrt((x1-pos[0])**2 + (y1-pos[1])**2 + (z1-pos[2])**2)
def is_in_bounds(x, y):
    return distance(x, y) < max_radius
def is_in_bounds_3d(pos, limit):
    x1 = 0; y1 = 0; z1 = target_z
    return np.sqrt((x1-pos[0])**2 + (y1-pos[1])**2 + (z1-pos[2])**2) < limit
def loadweights(type): 
    if type == 1:
        f = open('weights_output.txt', 'r')
        return np.array([ list(map(np.float32,line.split())) for line in f ])
    else: 
        f = open('weights_hidden.txt', 'r')
        return np.array([ list(map(np.float32,line.split())) for line in f ])

def draw_rewards(reward_list, qlearning, block):
    plt.close()
    plt.subplot(2, 1, 1) # set to first column plot
    if qlearning:
        plt.title("Average Reward per Episode (Q-Learning)")
    else:
        plt.title("Average Reward per Episode (SARSA)")
    plt.xlabel("Episode number")
    plt.ylabel("Reward")
    plt.plot(reward_list, label="Reward")
    plt.legend()

    plt.subplot(2, 1, 2) # set to first column plot
    if qlearning:
        plt.title("Average Reward per 100 Episodes (Q-Learning)")
    else:
        plt.title("Average Reward per 100 Episodes (SARSA)")
    plt.xlabel("Episode number (100's)")
    plt.ylabel("Reward")
    avg = np.zeros(len(reward_list)//100 + 1)
    for index, val in enumerate(reward_list):
        avg[index//100] += val
    for i in range(len(avg)-1):
        avg[i] /= 100
    avg[len(avg)-1] /= len(reward_list) % 100
    plt.plot(avg, label="Reward")
    plt.legend()
    plt.tight_layout()
    plt.show(block=block)

# init drone
client = MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# hidden layer
inputs1 = tf.placeholder(shape=[1,num_inputs], dtype=tf.float32)
if readWeights:
    weights_hidden = tf.Variable(loadweights(0))
else:
    weights_hidden = tf.Variable(tf.random_normal([num_inputs, num_hidden]))
bias_hidden = tf.Variable(tf.random_normal([num_hidden]))
# preactivations_hidden = tf.add(tf.matmul(inputs1, weights_hidden), bias_hidden)
preactivations_hidden = tf.matmul(inputs1, weights_hidden)
# activations_hidden = tf.nn.sigmoid(preactivations_hidden)
activations_hidden = tf.tanh(preactivations_hidden)

# output layer
if readWeights:
    weights_output = tf.Variable(loadweights(1))
else:
    weights_output = tf.Variable(tf.random_normal([num_hidden, num_outputs]))
bias_output = tf.Variable(tf.random_normal([num_outputs]))
# Qout = tf.add(tf.matmul(activations_hidden, weights_output), bias_output)
Qout = tf.matmul(activations_hidden, weights_output)

predict = tf.argmax(Qout,1)

# training
nextQ = tf.placeholder(shape=[1,num_outputs], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()

#create lists to contain total rewards and steps per episode
total_reward_list = np.zeros(num_episodes)
steps_to_success = np.zeros(num_episodes)
percent_success_actions = np.zeros(num_episodes)
num_to_graph = 0

with tf.Session() as sess:
    sess.run(init)
    # episode loop
    for i in range(num_episodes):

        if msvcrt.kbhit():
            # script must be run from cmd.exe in order to register keypresses
            print("You pressed ", msvcrt.getch(), " so now i will quit.")
            break

        print("\n\n\nEPISODE " + str(i) + "\n\n\n")

        #Reset drone and get state
        init_orient = (0, 0, 0)
        print("===== Initial Orientation " + str(init_orient))
        client.simSetPose(Pose(Vector3r(0,0,target_z), 
            AirSimClientBase.toQuaternion(init_orient[0], init_orient[1], init_orient[2])), True)

        success_counter = 0
        num_success_actions = 0
        num_actions_taken = 0

        # action loop
        for j in range(episode_length):
            
            # get current state
            print("===== Action " + str(j))
            curr_pos = client.getPosition()
            curr_orient = client.getRollPitchYaw()
            curr_s = [curr_pos.x_val, curr_pos.y_val, curr_pos.z_val, 
                curr_orient[0], curr_orient[1], curr_orient[2]]
            scaled_curr_s = scale_pos(curr_s)

            print("  STATE " + str(curr_s))
            print("  ====== scaled s " + str(scaled_curr_s))

            if not is_in_bounds(curr_s[0], curr_s[1]):
                # drone has gone too far -- reset
                print("===== OUT OF BOUNDS")
                break

            # a_index index of max action, allQ all Q-vals for current state
            a_index,allQ = sess.run([predict,Qout],feed_dict={inputs1:scaled_curr_s})

            if j == 0: 
                sarsa_index = a_index

            if(qlearning):
                # decide next action (angle change relative to previous roll and pitch)
                if np.random.rand(1) < e:
                    # epsilon-greedy - random option
                    print("  !!!!!!!! EPSILON")
                    next_action = get_action(np.random.randint(0, num_outputs, dtype="int64"))
                else:
                    next_action = get_action(a_index[0]) 
            else: 
                # SARSA 
                next_action = get_action(sarsa_index[0])

            # calculate action input to AirSim
            roll_diff = np.asscalar(next_action[0])
            pitch_diff = np.asscalar(next_action[1])
            print("  ====== next action " + str(next_action))
            rpy = client.getRollPitchYaw()

            roll = rpy[0] + roll_diff
            pitch = rpy[1] + pitch_diff
            yaw = 0; duration = 0.5; sleep_time = 0.1


            print("  ====== moving to (" + str(roll*90) + " " + str(pitch*90) + ")")
            # take action
            client.moveByAngle(pitch, roll,  target_z, yaw, 0.1)
            # time.sleep(sleep_time) # wait for action to occur 

            # get next state and reward as result of action
            s1Position = client.getPosition()
            s1Orientation = client.getRollPitchYaw()
            s1 = [s1Position.x_val, s1Position.y_val, s1Position.z_val, s1Orientation[0], s1Orientation[1], s1Orientation[2]]
            scaled_s1 = scale_pos(s1)
            r = reward(s1)
            total_reward_list[i] += r
            print("  ==== Reward " + str(r))

            # evaluate goal criteria
            if did_reach_goal(s1):
                print("  ******* reached goal " )
                num_success_actions += 1
                success_counter += 1
                if success_counter >= 30:
                    print("\n\n SUCCESS " + str(i) + "\n\n")
                    # record number of steps to success
                    steps_to_success[i] = j
                    # break
            else:
                # make sure successful actions are consecutive
                success_counter = 0

            # Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:scaled_s1})
            
            if qlearning:
                # Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1) # from neural net
                print("  ===== MAX Q1 " + str(maxQ1))
                targetQ = allQ
                targetQ[0,a_index[0]] = r + y*maxQ1
                print("  ===== TARGET " + str(r + y*maxQ1))
            else:
                # SARSA
                if np.random.rand(1) < e:
                    sarsa_index[0] = np.random.randint(0, num_outputs)
                    # epsilon-greedy - random option
                    print("  !!!!!!!! EPSILON IN SARSA")
                else:
                    sarsa_index[0] = np.asscalar(np.argmax(Q1))
                actual_q = Q1[0][sarsa_index[0]]
                targetQ = allQ
                targetQ[0,sarsa_index[0]] = r + y*actual_q
                print("  ===== TARGET " + str(r + y*actual_q))     

            # train ANN using target Q
            _,W1,W0 = sess.run([updateModel,weights_output, weights_hidden ], feed_dict={inputs1:scaled_curr_s,nextQ:targetQ})

            with open("weights_output.txt", "w") as weights_file:
                weights_file.write(str(W1))
            with open("weights_hidden.txt", "w") as weights_file:
                weights_file.write(str(W0))

            num_actions_taken += 1

        # episode done 
        print("\n\n\nTotal Reward")
        print(total_reward_list[i])
        print(num_actions_taken)
        print("\n\n\n")

        total_reward_list[i] = total_reward_list[i]/num_actions_taken
        percent_success_actions[i] = num_success_actions/num_actions_taken
        e = 2./((i/1000) + 10)
        num_to_graph += 1
        if i % 50 == 0:
            draw_rewards(total_reward_list[:num_to_graph], qlearning, False)
        print("Epsilon " + str(e))
 
# print("WEIGHTS\n" + str(W1))
plt.close()
plt.title("Number of Actions Taken to Reach Goal")
plt.xlabel("Episode number")
plt.ylabel("Actions")
plt.plot(steps_to_success[:num_to_graph], label="Actions")
plt.legend()
plt.show()

plt.title("Percentage of Successful Actions Per Episode")
plt.xlabel("Episode number")
plt.ylabel("Percentage")
plt.plot(np.multiply(percent_success_actions[:num_to_graph],100.0), label="Percent")
plt.legend()
plt.show()

draw_rewards(total_reward_list[:num_to_graph], qlearning, True)
