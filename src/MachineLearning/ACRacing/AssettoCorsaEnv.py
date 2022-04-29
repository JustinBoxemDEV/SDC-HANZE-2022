# Steal from gym Carla implementation https://github.com/cjy1992/gym-carla/blob/master/gym_carla/envs/carla_env.py
# Brainstorm version

# I DO NOT KNOW IF THIS WORKS

from time import sleep
import gym
from gym import spaces
import numpy as np
import socket
import struct
import pyautogui

CAN_MSG_SENDING_SPEED = .040 # 25Hz
IP = "127.0.0.1"
PORT = 5454

ushort_to_bytes = struct.Struct('>H').pack
float_to_bytes = struct.Struct('f').pack

# helper functions
def shortkey(key2, key1='ctrlright'):
    pyautogui.keyDown(key1)
    pyautogui.keyDown(key2)
    pyautogui.keyUp(key1)
    pyautogui.keyUp(key2)
    
def reset_pos():
    sleep(2) 
    shortkey('o')
    sleep(2) # sometimes it wont register the key if it happens too fast
    shortkey('y')

def get_current_frame():
    frame = None
    return frame


class AssettoCorsaEnv(gym.Env):
    def __init__(self):
        print("Assetto Corsa Environment")
        self.display_size = 500
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.action_space = spaces.Box(
            np.array([-1, 0, 0]).astype(np.float32),
            np.array([+1, +1, +1]).astype(np.float32),
            # steer and gas, could add brake here
        )
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.display_size, self.display_size, 3), dtype=np.uint8) # not sure yet
    
    def reset(self):
        # Reset AC (go to starting position)
        reset_pos()
        # Set gear to 1 again
        self.client_socket.sendto(ushort_to_bytes(0x121) + bytes([0]*8), (IP, PORT))

        obversation = get_current_frame() # Screen capture
        return obversation

    def step(self, action):
        steer = action[0]
        acc = action[1]

        # print("Steering with", steer)
        # print("Acceleterating with", acc)
        
        # send steer and acc on socket to controller.py
        self.client_socket.sendto(ushort_to_bytes(0x12c) + float_to_bytes(steer) + bytes([0]*4), (IP, PORT))
        self.client_socket.sendto(ushort_to_bytes(0x120) + bytes([round(acc * 100)] + [0]*7), (IP, PORT))
        # self.client_socket.sendto(ushort_to_bytes(0x126) + bytes([round(brake * 100)] + [0]*7), (IP, PORT)) # we do not use brake
        
        observation = get_current_frame() # Screen capture
        reward = self.get_reward(observation)
    
        done = False
        info = {}

        # Returning mandatory gym values
        return observation, reward, done, info

    def render(self):
        # Don't need this i think?
        pass

    def get_reward(self):
        
        # Negative points for driving too slow
        # Negative points for driving on grass (green pixels)
        # Positive points for driving on the track (grey pixels)
        # Positive points for finishing track faster than previous time
        reward = 0
        return reward
