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

def setHSV(frame, blur):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if(blur):
        hsv = cv2.medianBlur(hsv, 1)
    hsv = np.array(hsv)
    return hsv

def setMask(hsv, lower, upper):
    lower = np.array(lower)
    upper = np.array(upper)

    mask = cv2.inRange(hsv, lower, upper)
    return mask

def setColor(frame, mask, color):
    print(mask>0)
    frame[mask>0]=color
    return frame

# frame = cv2.imread("TestImges/1.png")# 480p image for testing
frame = cv2.imread("D:\\Github Desktop Clones\\SDC-HANZE-2022\\assets\\images\\assen.png") # 480p image for testing
frame = frame[30:510, 10:650] # 480p

# assetto to IRL conversion

# hsv = setHSV(frame, False)
# mask = setMask(hsv, [18, 90, 40], [41, 145, 80]) # set mask for Assetto grass
# frame = setColor(frame, mask, (35, 120, 100)) # set assetto grass to real life grass color

# hsv = setHSV(frame, False)
# mask = setMask(hsv, [0, 0, 0], [25, 100, 150]) # set mask for Assetto road
# frame = setColor(frame, mask, (100, 81, 82)) # set assetto road to real life road color

# hsv = setHSV(frame, False)
# mask = setMask(hsv, [100, 0, 167], [179, 255, 224]) # set white pixels
# cv2.imshow("white pixels", mask)
# frame = setColor(frame, mask, (359, 0, 100)) # set assetto road to real life road color

cv2.imshow("Assetto TO IRL", frame)

# set False if converting assetto to IRL, set True if using IRL image
hsv = setHSV(frame, True)

# final grass mask
# mask = setMask(hsv, [23, 0, 0], [42, 255, 191]) # IRL grass mask

# final road mask
mask = setMask(hsv, [95, 0, 50], [155, 80, 150]) # IRL road mask

cv2.imshow("Final Mask", mask)

# ACWindow = win32gui.FindWindow(None, "Assetto Corsa")
# rect = win32gui.GetWindowPlacement(ACWindow)[-1]
# frame = np.array(ImageGrab.grab(rect))[:,:,::-1]

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

def nothing(x):
    pass
tweak = cv2.imread("D:\\Github Desktop Clones\\SDC-HANZE-2022\\assets\\images\\assen.png")
# tweak = frame.copy()
# tweak = cv2.medianBlur(tweak, 27)

# Create a window
cv2.namedWindow('image')

# Create trackbars for color change
# Hue is from 0-179 for Opencv
cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

# Set default value for Max HSV trackbars
cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# Initialize HSV min/max values
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

while(1):
    # Get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin', 'image')
    sMin = cv2.getTrackbarPos('SMin', 'image')
    vMin = cv2.getTrackbarPos('VMin', 'image')
    hMax = cv2.getTrackbarPos('HMax', 'image')
    sMax = cv2.getTrackbarPos('SMax', 'image')
    vMax = cv2.getTrackbarPos('VMax', 'image')

    # Set minimum and maximum HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(tweak, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(tweak, tweak, mask=mask)

    # Print if there is a change in HSV value
    if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display result image
    cv2.imshow('image', result)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
