# TEST FILE WITH SHIT

# ------------------------------------------------------------------------------------
# count green pixels in region of interest (roi)
import cv2
from cv2 import imread
# import win32gui
#from PIL import ImageGrab
import numpy as np
import cv2

# ACWindow = win32gui.FindWindow(None, "Assetto Corsa")
# rect = win32gui.GetWindowPlacement(ACWindow)[-1]
# frame = np.array(ImageGrab.grab(rect))[:,:,::-1]
# cv2.imshow("frame", frame)

# hsv = hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv, [18, 90, 40], [41, 145, 70])
# cv2.imshow("mask", mask)

# roi = mask[400:420, 130:530]
# cv2.imshow("roi", roi)

# green_pixels = 0
# for row in roi:
#     for pixel in row:
#         if pixel == 255:
#             green_pixels = green_pixels + 1
#             # print("woo")

def setHSV(frame, blur):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if(blur != 0):
        hsv = cv2.medianBlur(hsv, blur)
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

# frame = cv2.imread("src/MachineLearning/ACRacing/TestImges/ac480p.png")# 480p image for testing
# frame = cv2.imread("src/MachineLearning/ACRacing/TestImges/ac480pgrass.png")# 480p image for testing
frame = cv2.imread("src/MachineLearning/ACRacing/TestImges/ac480phalfgrass.png")# 480p image for testing
# frame = cv2.imread("/home/douwe/Projects/SDC-HANZE-2022/assets/images/assen.png") # 480p image for testing
frame = frame[30:510, 10:650] # 480p

cv2.imshow("Original frame", frame)

# assetto to IRL conversion

tweak = frame.copy();

hsv = setHSV(frame, 0)
mask = setMask(hsv, [0,43, 97], [28, 68, 159])
frame = setColor(frame, mask, (255, 0, 0))

hsv = setHSV(frame, 0)
mask = setMask(hsv, [24, 74, 0], [70, 255, 255]) # set mask for Assetto grass
frame = setColor(frame, mask, (35, 120, 100)) # set assetto grass to real life grass color

hsv = setHSV(frame, 0)
mask = setMask(hsv, [0, 0, 0], [25, 100, 150]) # set mask for Assetto road
frame = setColor(frame, mask, (100, 81, 82)) # set assetto road to real life road color

# tweak = frame.copy()
cv2.imshow("Assetto TO IRL", frame)

# set False if converting assetto to IRL, set True if using IRL image
hsv = setHSV(frame, 0)

# final grass mask
# mask = setMask(hsv, [23, 0, 0], [42, 255, 191]) # IRL grass mask

# final road mask
mask = setMask(hsv, [98, 0, 0], [179, 82, 150]) # IRL road mask

cv2.imshow("Final Mask", mask)

mask = cv2.medianBlur(mask, 3)

cv2.imshow("Final mask with blur", mask)

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

def nothing(x):
    pass

# tweak = cv2.imread("TestImges/ac480p.png")
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
