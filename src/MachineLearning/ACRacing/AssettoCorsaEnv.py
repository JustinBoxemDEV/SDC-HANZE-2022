# Inspired by gym Carla implementation https://github.com/cjy1992/gym-carla/blob/master/gym_carla/envs/carla_env.py

from itertools import count
from time import sleep
import time
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
import time
if platform == "win32":
    import win32gui
from PIL import ImageGrab

CAN_MSG_SENDING_SPEED = .040 # 25Hz
IP = "127.0.0.1"
PORT = 5454

ushort_to_bytes = struct.Struct('>H').pack
float_to_bytes = struct.Struct('f').pack

# helper functions
def setHSV(frame, blur):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if(blur):
        hsv = cv2.medianBlur(hsv, 27)
    hsv = np.array(hsv)
    return hsv

def setMask(hsv, lower, upper):
    lower = np.array(lower)
    upper = np.array(upper)

    mask = cv2.inRange(hsv, lower, upper)
    return mask

def setColor(frame, mask, color):
    frame[mask>0]=color
    return frame

def shortkey(key2, key1='ctrlright'):
    pyautogui.keyDown(key1)
    pyautogui.keyDown(key2)
    pyautogui.keyUp(key1)
    pyautogui.keyUp(key2)
    
def reset_pos():
    shortkey('o')
    sleep(2) # sometimes it wont register the shortkey if it happens too fast
    shortkey('y')

def get_current_observation():
    ACWindow = win32gui.FindWindow(None, "Assetto Corsa")
    rect = win32gui.GetWindowPlacement(ACWindow)[-1]
    frame = np.array(ImageGrab.grab(rect))[:,:,::-1]

    # assetto to IRL conversion
    # hsv = setHSV(frame, True)
    # mask = setMask(hsv, [18, 90, 40], [41, 145, 70]) # set mask for Assetto grass
    # frame = setColor(frame, mask, (35, 120, 100)) # set assetto grass to real life grass color

    # hsv = setHSV(frame, True)
    # mask = setMask(hsv, [0, 0, 0], [25, 100, 150]) # set mask for Assetto road
    # frame = setColor(frame, mask, (100, 81, 82)) # set assetto road to real life road color
    
    # frame = frame[30:510, 10:650] # 480p, cut a couple pixels to fit the model and screen
    roi = frame[380:420, 50:600] # roi in 480p AC image
    
    return roi

def count_pixels(observation, lower_range, upper_range):
    """Counts the amount of pixels within the colour range lower_range and upper_range (HSV)
    :param observation The image in which the pixels will be counted
    :param lower_range The lower threshold as an array with 3 values (HSV format)
    :param upper_range The upper threshold as an array with 3 values (HSV format)
    """
    hsv = cv2.cvtColor(observation, cv2.COLOR_BGR2HSV)
    hsv = cv2.medianBlur(hsv, 7)

    # Mask in range
    lower_range = np.array(lower_range)
    upper_range= np.array(upper_range)
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # roi = mask[400:420, 130:530] # only works for 480p AC image
    # roi = mask[380:420, 50:600]

    pixels_amt = 0
    for row in mask:
        for pixel in row:
            if pixel == 255:
                pixels_amt = pixels_amt + 1

    return pixels_amt
 
class AssettoCorsaEnv(gym.Env):
    def __init__(self):
        print("Assetto Corsa Environment")

        # 480p image
        # self.display_height = 480
        # self.display_width = 640

        # roi
        self.display_height = 40
        self.display_width = 550

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

        obversation = get_current_observation()

        # start timer for consecutive time spent on road
        self.epoch_time = int(time.time())

        return obversation

    def step(self, action):
        steer = action[0]
        # acc = action[1]

        acc = 1 # we ignore acceleration for now and only focus on steering

        print("Steering with", steer)
        # print("Accelerating with", acc)
        
        # send steer and acc to controller.py via socket
        self.client_socket.sendto(ushort_to_bytes(0x12c) + float_to_bytes(steer) + bytes([0]*4), (IP, PORT))
        self.client_socket.sendto(ushort_to_bytes(0x120) + bytes([round(acc * 100)] + [0]*7), (IP, PORT))
        # self.client_socket.sendto(ushort_to_bytes(0x126) + bytes([round(brake * 100)] + [0]*7), (IP, PORT)) # we do not use brake
        
        observation = get_current_observation()

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

        green_pixels = count_pixels(observation, [24, 74, 0], [70, 255, 255])
        # print(f"Grass pixels: {green_pixels}")

        # Negative points for driving on grass (green pixels)
        if green_pixels > 10000:
            reward = reward - 50
            done = True
            # print("Too many green pixels,", green_pixels,". restarting.")
        elif green_pixels > 4000:
            reward = reward - 20
            done = False
        elif green_pixels > 50:
            reward = reward - 10
            done = False
        elif green_pixels > 3: # small error offset for rogue pixels, should be 0
            reward = reward - 1
            done = False
        else:
            # Nothing
            done = False
        
        road_pixels = count_pixels(observation, [0, 0, 0], [25, 100, 150])
        # print(f"Road pixels: {road_pixels}")

        # I dont think we need this anymore?
        # if road_pixels > 5000: # TODO: tweak these values
        #     reward = reward + 2
        # elif road_pixels > 1000: # TODO: tweak these values
        #     reward = reward + 1
        # else:
        #     reward = reward - 20
        
        # Negative points for driving too slow (currently not used because we do not use acceleration)
        # if acceleration > 0.9:
        #     reward = reward + 1
        # elif acceleration == 0:
        #     reward = reward - 10
        # else:
        #     reward = reward - 5

        # Time spent on road
        if green_pixels > 3 or road_pixels < 10000: # small error offset for rogue pixels, should be 0
            consecutive_time_spent_on_road = int(time.time()) - self.epoch_time 
            
            if consecutive_time_spent_on_road < 10: 
                print(f"Not enough time spent on the road: {consecutive_time_spent_on_road}")
                # reward = reward - 1
            elif consecutive_time_spent_on_road > 20:
                print(f"more than 20 seconds on the road, +2: {consecutive_time_spent_on_road}")
                reward = reward + 2
            else:
                print(f"5 or more seconds on the road, +1: {consecutive_time_spent_on_road}")
                reward = reward + 1

            self.epoch_time = int(time.time())
        else:
            # reward = reward + 1
            pass

        # Negative points for driving over the white line!
        white_pixels = count_pixels(observation, [0,43, 97], [28, 68, 159])
        if white_pixels > 300:
            reward = reward - 20
            # done = True
        elif white_pixels > 250:
            reward = reward - 10
        else:
            # Nothing
            pass

        return reward, done

    def close(self):
        print("Closing...")
        sys.exit()
