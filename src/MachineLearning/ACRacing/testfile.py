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
# from AssettoCorsaEnv import AssettoCorsaEnv
# torch.cuda.empty_cache()
# env = AssettoCorsaEnv()
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
# import cv2
# from cv2 import imread
# img_name_road = "C:/Users/Sabin/Documents/SDC/Screenshot 2022-04-29 193118.png"
# img_name_gras = "C:/Users/Sabin/Documents/SDC/Screenshot 2022-04-29 200655.png"
# img_name_little_grass = "C:/Users/Sabin/Documents/SDC/Screenshot 2022-04-29 210018.png"
# img_name__very_little_grass = "C:/Users/Sabin/Documents/SDC/Screenshot 2022-04-29 210226.png"

# img_name_720p = "C:/Users/Sabin/Documents/SDC/Screenshot 2022-04-29 215205_720p.png"

# img_road = imread(img_name_road)
# img_grass = imread(img_name_gras)
# img_little_grass = imread(img_name_little_grass)
# img_very_little_grass = imread(img_name__very_little_grass)

# img_720p = imread("C:/Users/Sabin/Documents/SDC/Screenshot 2022-04-29 215205_720p.png")

# # these values are on 1440p you will need to change them if you use this
# roi_road = img_road[1250:1280, 870:1709]
# roi_grass = img_grass[1250:1280, 870:1709]
# roi_little_grass = img_little_grass[1250:1280, 870:1709]
# roi_very_little_grass = img_very_little_grass[1250:1280, 870:1709]

# roi_720p = img_720p[576:606, 315:1090]

# cv2.imshow("roi road", roi_road)
# cv2.imshow("roi grass", roi_grass)
# cv2.imshow("roi little grass", roi_little_grass)
# cv2.imshow("roi very little grass", roi_very_little_grass)
# cv2.imshow("roi 720p", roi_720p)

# # USES BGR INSTEAD OF RGB
# # count_grass_pixels1 = 0
# # for row in roi_road:
# #     for pixel in row:
# #         # print(pixel)
# #         if pixel[1]-10 > pixel[0] and pixel[1]-10 > pixel[2]:
# #             count_grass_pixels1 = count_grass_pixels1 + 1
# # print("Grass pixels in image with only road:", count_grass_pixels1)

# # count_grass_pixels2 = 0
# # for row in roi_very_little_grass:
# #     for pixel in row:
# #         # print(pixel)
# #         if pixel[1]-10 > pixel[0] and pixel[1]-10 > pixel[2]:
# #             count_grass_pixels2 = count_grass_pixels2 + 1

# # print("Grass pixels in image with very little grass:", count_grass_pixels2)

# cv2.waitKey(0)

# -------------------------------------------------------------------------------------------------
# import win32gui
from PIL import ImageGrab
import numpy as np
import cv2

# ACWindow = win32gui.FindWindow(None, "Assetto Corsa")
# rect = win32gui.GetWindowPlacement(ACWindow)[-1]
# frame = np.array(ImageGrab.grab(rect))[:,:,::-1]
# frame = frame[:720, :1280]

frame = cv2.imread("/home/sab/Documents/Projects/SDC-HANZE-2022/src/MachineLearning/ACRacing/TestImges/ac480p.png") # 480p image
# frame = frame[:100, :100]

cv2.imshow("a", frame)

roi = frame[410:430, 180:585]

cv2.imshow("roi", roi)
cv2.waitKey(0)
