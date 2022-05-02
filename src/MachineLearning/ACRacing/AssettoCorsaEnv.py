# Steal from gym Carla implementation https://github.com/cjy1992/gym-carla/blob/master/gym_carla/envs/carla_env.py
# Brainstorm version

# I DO NOT KNOW IF THIS WORKS

from itertools import count
from time import sleep
from cv2 import imread
import gym
from gym import spaces
import numpy as np
import socket
import struct
import pyautogui
import random
import win32gui
from PIL import ImageGrab


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
    shortkey('o')
    sleep(2) # sometimes it wont register the shortkey if it happens too fast
    shortkey('y')

def get_current_frame():
    ACWindow = win32gui.FindWindow(None, "Assetto Corsa")
    rect = win32gui.GetWindowPlacement(ACWindow)[-1]
    frame = np.array(ImageGrab.grab(rect))[:,:,::-1]
    # frame = frame[:720, :1280] # 720p cut a couple pixels to fit the model
    frame = frame[:480, :720] # 480p
 
    # For testing
    # frame = imread("C:/Users/Sabin/Documents/SDC/Screenshot 2022-04-29 193118.png") # only road
    # frame = imread("C:/Users/Sabin/Documents/SDC/Screenshot 2022-04-29 210226.png") # some grass
    # frame = imread("C:/Users/Sabin/Documents/SDC/Screenshot 2022-04-29 212150.png") # only grass
    # frame = imread("C:/Users/Sabin/Documents/SDC/Screenshot 2022-04-29 215205_720p.png") # 720p image
    return frame

def count_green_pixels_ish(observation):
    # roi = observation[1250:1280, 870:1709] # 1440p
    # roi = observation[576:606, 315:1090] # 720p
    roi = observation[410:430, 180:585] # 480p
    
    grass_pixels_count = 0
    for row in roi:
        for pixel in row:
            if pixel[1]-10 > pixel[0] and pixel[1]-10 > pixel[2]:
                grass_pixels_count = grass_pixels_count + 1
    # print("Grass pixels(ish) in roi:", grass_pixels_count)

    return grass_pixels_count


class AssettoCorsaEnv(gym.Env):
    def __init__(self):
        print("Assetto Corsa Environment")
        # self.display_height = 720
        # self.display_width = 1280

        self.display_height = 480
        self.display_width = 720

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.action_space = spaces.Box(
            np.array([-1, 0, 0]).astype(np.float32),
            np.array([+1, +1, +1]).astype(np.float32),
            # steer and gas, could add brake here
        )
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.display_height, self.display_width, 3), dtype=np.uint8)
    
    def reset(self):
        # Reset AC (go to starting position)
        reset_pos()
        # Set gear to 1 again
        self.client_socket.sendto(ushort_to_bytes(0x121) + bytes([0]*8), (IP, PORT))

        obversation = get_current_frame()
        return obversation

    def step(self, action):
        steer = action[0]
        acc = action[1]

        print("Steering with", steer)
        print("Accelerating with", acc)
        
        # send steer and acc on socket to controller.py
        self.client_socket.sendto(ushort_to_bytes(0x12c) + float_to_bytes(steer) + bytes([0]*4), (IP, PORT))
        self.client_socket.sendto(ushort_to_bytes(0x120) + bytes([round(acc * 100)] + [0]*7), (IP, PORT))
        # self.client_socket.sendto(ushort_to_bytes(0x126) + bytes([round(brake * 100)] + [0]*7), (IP, PORT)) # we do not use brake
        
        observation = get_current_frame()
    
        # Temporary random condition to terminate, maybe a timer instead
        green_pixels = count_green_pixels_ish(observation)

        
        if green_pixels > 200:
            done = True
            print("Too many green pixels:", green_pixels)
        else:
            done = False
            print("Green pixels:", green_pixels)
        info = {}
        reward = self.get_reward(green_pixels)
        
        print("reward:", reward)
        # Returning mandatory gym values
        return observation, reward, done, info

    def render(self):
        # Don't need this i think?
        pass

    def get_reward(self, green_pixels):
        # check reward given current observation

        # SCUFFED
        # Negative points for driving on grass (green pixels)
        # Positive points for driving on the track (grey pixels)
        if green_pixels < 7:
            print("No grass here!")
            reward =+ 10
        else:
            print("We seem to be in the grass!")
            reward =- 50
        
        # Negative points for driving too slow
        # Positive points for finishing track faster than previous time
        return reward
