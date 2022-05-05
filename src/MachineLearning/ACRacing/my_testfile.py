import cv2
from cv2 import imread
import win32gui
from PIL import ImageGrab
import numpy as np
import cv2

def count_pixels(observation, lower_range, upper_range):
    """ Counts the amount of pixels within the colour range lower_range and upper_range (HSV)
    :param observation The image in which the pixels will be counted
    :param lower_range The lower threshold as an array with 3 values (HSV format)
    :param upper_range The upper threshold as an array with 3 values (HSV format)
    """

    cv2.imshow("frame", observation)

    hsv = cv2.cvtColor(observation, cv2.COLOR_BGR2HSV)

    # Mask in range
    lower_range = np.array(lower_range)
    upper_range= np.array(upper_range)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    cv2.imshow("mask", mask)

    roi = mask[400:420, 130:530] # only works for 480p AC image
    cv2.imshow("roi", roi)

    pixels_amt = 0
    for row in roi:
        for pixel in row:
            if pixel == 255:
                pixels_amt = pixels_amt + 1

    cv2.waitKey(0)
    return pixels_amt

ACWindow = win32gui.FindWindow(None, "Assetto Corsa")
rect = win32gui.GetWindowPlacement(ACWindow)[-1]
frame = np.array(ImageGrab.grab(rect))[:,:,::-1]

# 480p image for testing
# frame = cv2.imread("src/MachineLearning/ACRacing/TestImges/ac480p.png")
frame = cv2.imread("src/MachineLearning/ACRacing/TestImges/ac480pgrass.png")
# frame = cv2.imread("src/MachineLearning/ACRacing/TestImges/ac480phalfgrass.png")
frame = frame[30:510, 10:650]

# print(count_pixels(frame, [18, 90, 40], [41, 145, 70])) # green
print(count_pixels(frame, [0, 0, 0], [25, 100, 150])) # brown ish