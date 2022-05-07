# TEST FILE WITH SHIT

# ------------------------------------------------------------------------------------
# count green pixels in region of interest (roi)
from xml.etree.ElementTree import tostring
import cv2
import os
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
def nothing(x):
    pass

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

assetto = False
road = False
grass = False
lines = True

if(assetto):
    dir = "/home/douwe/Projects/SDC-HANZE-2022/src/MachineLearning/ACRacing/TestImges/assetto"
else:
    dir = "/home/douwe/Projects/SDC-HANZE-2022/src/MachineLearning/ACRacing/TestImges/real life"

tweak = cv2.imread("/home/douwe/Projects/SDC-HANZE-2022/src/MachineLearning/ACRacing/TestImges/real life/assen2.png")

width = int(tweak.shape[1] * 60 / 100)
height = int(tweak.shape[0] * 60 / 100)
dim = (width, height)

tweak = cv2.resize(tweak, dim, interpolation= cv2.INTER_AREA)

for subdir, dirs, files in os.walk(dir):
    for file in files:
        print(file)
        frame = cv2.imread(dir+"/"+file)
        
        originalFrame = frame.copy()
        
        # frame = frame[30:510, 10:650] # 480p

        if(assetto):
            # assetto to IRL conversion
            
            if(lines):
                hsv = setHSV(frame, 3)
                mask = setMask(hsv, [0,40, 124], [77, 61, 162]) # set mask for Assetto lines
                frame = setColor(frame, mask, (200, 200, 200)) # set assetto lines to real life lines color
                mask= cv2.medianBlur(mask, 11)
                frame = setColor(frame, mask, (192, 192, 200))
                cv2.imshow("white lines", mask)

            if(grass):
                hsv = setHSV(frame, 0)
                mask = setMask(hsv, [24, 74, 0], [70, 255, 255]) # set mask for Assetto grass
                frame = setColor(frame, mask, (35, 120, 100)) # set assetto grass to real life grass color
            
            if(road):
                hsv = setHSV(frame, 7)
                mask = setMask(hsv, [0, 0, 0], [25, 255, 74]) # set mask for Assetto road
                frame = setColor(frame, mask, (78, 62, 63)) # set assetto road to real life road color
        
            # tweak = frame.copy()
            
            AssettoToIRLFrame = frame.copy()
            
            # cv2.imshow(file+"Assetto TO IRL", frame)
            hsv = setHSV(frame, 0)

        if(assetto == False):
            hsv = setHSV(frame, 7)
            
        # final grass mask
        if(grass):
            mask = setMask(hsv, [23, 0, 0], [42, 255, 191]) # IRL grass mask
            mask = cv2.medianBlur(mask, 7)
        
        # cv2.imshow(file+"hsv mask", mask)
        
        # final road mask
        if(road):
            mask = setMask(hsv, [0, 0, 0], [179, 65, 165]) # IRL road mask

        # final line mask
        if(lines):
            mask = setMask(hsv, [0, 0, 170], [179, 72, 255]) # IRL line mask
        
        unblurredMask = mask.copy()
        
        mask = cv2.medianBlur(mask, 7)
        blurredMask = mask.copy()
        
        roi = mask[380:420, 50:600] # only works for 480p AC image
        
        cv2.imshow(file+" original image", originalFrame)
        
        if(assetto):
            cv2.imshow(file+" Assetto to IRL", AssettoToIRLFrame)
        cv2.imshow(file+" unblurred mask", unblurredMask)
        cv2.imshow(file+" blurred mask", blurredMask)
        cv2.imshow(file+"roi", roi)
        
        # ACWindow = win32gui.FindWindow(None, "Assetto Corsa")
        # rect = win32gui.GetWindowPlacement(ACWindow)[-1]
        # frame = np.array(ImageGrab.grab(rect))[:,:,::-1]

        pixels_amt = 0
        for row in roi:
            for pixel in row:
                if pixel == 255:
                    pixels_amt = pixels_amt + 1
                    
        print("roi pixels:", pixels_amt)

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
