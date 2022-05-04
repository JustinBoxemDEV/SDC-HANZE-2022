# TEST FILE WITH SHIT

# Random values test
# from AssettoCorsaEnv import AssettoCorsaEnv

# env = AssettoCorsaEnv()
# # env.reset()
# # env.step()

# episodes = 5
# for episode in range(episodes):
#     observation = env.reset()
#     done = False
#     score = 0

#     while not done:
#         action = env.action_space.sample() # random actions
#         observation, reward, done, _ = env.step(action)
#         score += reward
#     print("Episode: {} Score: {}".format(episode, score))

# ------------------------------------------------------------------------------------
# Test model
# import torch
# import os
# from AssettoCorsaEnv import AssettoCorsaEnv
# from stable_baselines3 import PPO

# save_path = os.path.join("src/MachineLearning/ACRacing/", "training", "models", "AC_model_3")

# torch.cuda.empty_cache()
# env = AssettoCorsaEnv()
# model = PPO.load(save_path, env=env, device='cuda')

# print('testing...')
# episodes = 5
# for episode in range(episodes):
#     observation = env.reset()
#     done = False
#     score = 0

#     while not done:
#         # env.render()
#         action = model.predict(observation)
#         # print(f"action: {action}")
#         observation, reward, done, _ = env.step(action[0])
#         score += reward
#     print("Episode: {} Score: {}".format(episode, score))
# env.close()
    

# ------------------------------------------------------------------------------------
# count green pixels in region of interest (roi)
import cv2
from cv2 import imread
import win32gui
from PIL import ImageGrab
import numpy as np
import cv2

# ACWindow = win32gui.FindWindow(None, "Assetto Corsa")
# rect = win32gui.GetWindowPlacement(ACWindow)[-1]
# frame = np.array(ImageGrab.grab(rect))[:,:,::-1]

frame = cv2.imread("src/MachineLearning/ACRacing/TestImges/ac480p.pn") # 480p image for testing
# frame = frame[30:510, 10:650] # 480p
# cv2.imshow("img", frame)

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

cv2.imshow("hsv", hsv)

hsv = cv2.medianBlur(hsv, 15)

cv2.imshow("blur", hsv)

# range green
lower_green = np.array([32, 0, 0])
print(lower_green)
upper_green= np.array([55, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)

cv2.imshow("mask", mask)

roi = mask[400:420, 130:530]
cv2.imshow("roi", roi)

green_pixels = 0
for row in roi:
    for pixel in row:
        if pixel == 255:
            green_pixels = green_pixels + 1
            # print("woo")

print("Green pixels:", green_pixels)

# USES BGR INSTEAD OF RGB
# count_grass_pixels1 = 0
# for row in roi_road:
#     for pixel in row:
#         # print(pixel)
#         if pixel[1]-10 > pixel[0] and pixel[1]-10 > pixel[2]:
#             count_grass_pixels1 = count_grass_pixels1 + 1
# print("Grass pixels in image with only road:", count_grass_pixels1)

cv2.waitKey(0)