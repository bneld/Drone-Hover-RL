import numpy as np
import random
import matplotlib.pyplot as plt
from AirSimClient import *
import sys
import time
import random
import msvcrt

np.set_printoptions(threshold=np.nan)

# drone boundaries
goal_limit = 0.5
max_radius = 3

target_z = -2 # target height in NED coordinate system
num_episodes = 20000
episode_length = 100 # number of actions per episode

def reward(state):
    # if did_reach_goal(state):
    if is_in_bounds_3d(state[:3], goal_limit):
        return 10
    else:
        return -50
def did_reach_goal(state):
    return is_in_bounds_3d(state[:3], goal_limit)
# takes an index of the action space (max Q) and converts to the action values a = (roll, pitch)

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

def draw_rewards(reward_list, block):
    plt.close()
    plt.subplot(2, 1, 1) # set to first column plot
    plt.title("Average Reward per Episode (Heuristic)")
    plt.xlabel("Episode number")
    plt.ylabel("Reward")
    plt.plot(reward_list, label="Reward")
    plt.legend()

    plt.subplot(2, 1, 2) # set to first column plot
    plt.title("Average Reward per 100 Episodes (Heuristic)")
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

#create lists to contain total rewards and steps per episode
total_reward_list = np.zeros(num_episodes)
steps_to_success = np.zeros(num_episodes)
percent_success_actions = np.zeros(num_episodes)
num_to_graph = 0

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

        print("  STATE " + str(curr_s))

        if not is_in_bounds(curr_s[0], curr_s[1]):
            # drone has gone too far -- reset
            print("===== OUT OF BOUNDS")
            break

        # use heuristic to calculate action input to AirSim
        rpy = client.getRollPitchYaw()
        angle_delta = 0.1
        roll_diff = ( -angle_delta if curr_orient[0] > 0 else angle_delta)
        pitch_diff = (-angle_delta if curr_orient[1] > 0 else angle_delta)
        roll = curr_orient[0] + roll_diff
        pitch = curr_orient[1] + pitch_diff
        yaw = 0; duration = 0.5; sleep_time = 0.1


        print("  ====== moving to (" + str(roll*90) + " " + str(pitch*90) + ")")
        # take action
        client.moveByAngle(pitch, roll,  target_z, yaw, 0.1)
        # time.sleep(sleep_time) # wait for action to occur 

        # get next state and reward as result of action
        s1Position = client.getPosition()
        s1Orientation = client.getRollPitchYaw()
        s1 = [s1Position.x_val, s1Position.y_val, s1Position.z_val, s1Orientation[0], s1Orientation[1], s1Orientation[2]]
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
        else:
            # make sure successful actions are consecutive
            success_counter = 0

        num_actions_taken += 1

    # episode done 
    print("\n\n\nTotal Reward")
    print(total_reward_list[i])
    print(num_actions_taken)
    print("\n\n\n")

    total_reward_list[i] = total_reward_list[i]/num_actions_taken
    percent_success_actions[i] = num_success_actions/num_actions_taken
    num_to_graph += 1
    if i % 50 == 0:
        draw_rewards(total_reward_list[:num_to_graph], False)
 
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

draw_rewards(total_reward_list[:num_to_graph], True)
