import cv2
from cv2 import imread
import win32gui
from PIL import ImageGrab
import numpy as np
import cv2
import time

def count_pixels(observation, lower_range, upper_range):
    """ Counts the amount of pixels within the colour range lower_range and upper_range (HSV)
    :param observation The image in which the pixels will be counted
    :param lower_range The lower threshold as an array with 3 values (HSV format)
    :param upper_range The upper threshold as an array with 3 values (HSV format)
    """

    cv2.imshow("frame", observation)

    hsv = cv2.cvtColor(observation, cv2.COLOR_BGR2HSV)
    hsv = cv2.medianBlur(hsv, 7)
    
    # Mask in range
    lower_range = np.array(lower_range)
    upper_range= np.array(upper_range)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    cv2.imshow("White pixels mask", mask)

    # roi = mask[400:420, 130:530] # only works for 480p AC image
    roi = mask[380:420, 50:600] # only works for 480p AC image
    cv2.imshow("roi", roi)

    pixels_amt = 0
    for row in roi:
        for pixel in row:
            if pixel == 255:
                pixels_amt = pixels_amt + 1

    cv2.waitKey(0)
    return pixels_amt

# ACWindow = win32gui.FindWindow(None, "Assetto Corsa")
# rect = win32gui.GetWindowPlacement(ACWindow)[-1]
# frame = np.array(ImageGrab.grab(rect))[:,:,::-1]

# 480p image for testing
# frame = cv2.imread("src/MachineLearning/ACRacing/TestImges/ac480p.png")
frame = cv2.imread("src/MachineLearning/ACRacing/TestImges/ac480p2.png")
# frame = cv2.imread("src/MachineLearning/ACRacing/TestImges/ac480p3.png")
# frame = cv2.imread("src/MachineLearning/ACRacing/TestImges/ac480pgrass.png")
# frame = cv2.imread("src/MachineLearning/ACRacing/TestImges/ac480phalfgrass.png")
# frame = cv2.imread("src/MachineLearning/ACRacing/TestImges/ac480pwhiteline.png")
# frame = cv2.imread("src/MachineLearning/ACRacing/TestImges/ac480pwhitecorner.png")
frame = frame[30:510, 10:650]

print("Grass pixels in roi:", count_pixels(frame, [23, 0, 0], [42, 255, 191])) # green
# print("Road pixels in roi:", count_pixels(frame, [0, 0, 0], [25, 100, 150])) # brown ish
# print("White pixels in roi:", count_pixels(frame, [0,43, 97], [28, 68, 159])) # white)

# -----------------------------------------------------------------------------------------------------
# first =int(time.time())
# print(first)
# time.sleep(5)
# second = int(time.time())
# print(second)

# third = second - first
# print(f"Seconds elapsed:{third}")

# -----------------------------------------------------------------------------------------------------
# For powerpoint images
# cv2.imshow("frame", frame)
# roi = frame[380:420, 50:600] # only works for 480p AC image
# cv2.imshow("roi", roi)
# cv2.waitKey(0)