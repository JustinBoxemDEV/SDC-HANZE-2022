# Inspired by gym Carla implementation https://github.com/cjy1992/gym-carla/blob/master/gym_carla/envs/carla_env.py

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
from sys import platform
import sys
import cv2
if platform == "win32":
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
    if platform == "win32" or platform =="cygwin":
        ACWindow = win32gui.FindWindow(None, "Assetto Corsa")
        rect = win32gui.GetWindowPlacement(ACWindow)[-1]
        frame = np.array(ImageGrab.grab(rect))[:,:,::-1]
        frame = frame[30:510, 10:650]# 480p cut a couple pixels to fit the model and screen
    else:
        # For testing on linux
        frame = imread("/src/MachineLearning/ACRacing/TestImges/ac480p.png") # 480p image
        frame = frame[:100, :100]

    return frame

def count_green_pixels_ish(observation):
    hsv = cv2.cvtColor(observation, cv2.COLOR_BGR2HSV)

    # range green (maybe tweak)
    lower_green = np.array([36, 0, 0])
    upper_green= np.array([86, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    roi = mask[400:420, 130:530] # only works for 480p AC image

    green_pixels = 0
    for row in roi:
        for pixel in row:
            if pixel == 255:
                green_pixels = green_pixels + 1

    print("Green pixels:", green_pixels)

    return green_pixels

class AssettoCorsaEnv(gym.Env):
    def __init__(self):
        print("Assetto Corsa Environment")

        if platform == "win32" or platform =="cygwin":
            self.display_height = 480
            self.display_width = 640
        else:
            self.display_height = 100
            self.display_width = 100

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.action_space = spaces.Box(
            # np.array([-1, 0]).astype(np.float32),
            # np.array([+1, +1]).astype(np.float32)
            # steer and gas, could add brake here

            # only steer
            np.array([-1]).astype(np.float32),
            np.array([+1]).astype(np.float32)
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
        # acc = action[1]

        acc = 1 # we ignore acceleration for now and only focus on steering

        print("Steering with", steer)
        print("Accelerating with", acc)
        
        # send steer and acc to controller.py via socket
        self.client_socket.sendto(ushort_to_bytes(0x12c) + float_to_bytes(steer) + bytes([0]*4), (IP, PORT))
        self.client_socket.sendto(ushort_to_bytes(0x120) + bytes([round(acc * 100)] + [0]*7), (IP, PORT))
        # self.client_socket.sendto(ushort_to_bytes(0x126) + bytes([round(brake * 100)] + [0]*7), (IP, PORT)) # we do not use brake
        
        observation = get_current_frame()

        reward, done = self.get_reward(observation)

        info = {}
        
        print("Total step reward:", reward)

        # Returning mandatory gym values
        return observation, reward, done, info

    def render(self, mode):
        # Don't need this due to Assetto Corsa rendering the game
        pass

    def get_reward(self, observation):
        # check reward given current observation
        reward = 0

        green_pixels = count_green_pixels_ish(observation)

        # Negative points for driving on grass (green pixels)
        # Positive points for driving on the track (grey pixels)
        if green_pixels > 100:
            reward = reward - 20
            done = True
            # print("Too many green pixels,", green_pixels,". restarting.")
        elif green_pixels > 50:
            reward = reward - 10
            done = False
        elif green_pixels > 15:
            reward = reward - 5
            done = False
        elif green_pixels > 7:
            reward = reward - 1
            done = False
        else:
            reward = reward + 4
            done = False
        
        # Negative points for driving too slow (currently not used because we do not use acceleration)
        # if acceleration > 0.9:
        #     reward = reward + 1
        # elif acceleration == 0:
        #     reward = reward - 10
        # else:
        #     reward = reward - 5

        return reward, done

    def close(self):
        print("Closing...")
        sys.exit()
