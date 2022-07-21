# Script for checking image loading speed
import cv2 as cv
import os
import time
start_time = time.time()

i = 0

for filename in os.listdir("C:/Users/Sabin/Documents/SDC/SL_data/images 24-05-2022 13-54-10/"):
  image = cv.imread("C:/Users/Sabin/Documents/SDC/SL_data/images 24-05-2022 13-54-10/"+filename)

print("--- %s seconds ---" % (time.time() - start_time))