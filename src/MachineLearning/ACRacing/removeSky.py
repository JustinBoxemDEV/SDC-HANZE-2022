import cv2
import os
import numpy as np
from screeninfo import get_monitors

height = 0
width = 0

for m in get_monitors():
    height = m.height
    width = m.width
    
print(height, width)

directory = os.fsencode("/home/douwe/Desktop/data/images 18-11-2021 15-12-21/")
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
   
    image = cv2.imread("/home/douwe/Desktop/data/images 18-11-2021 15-12-21/"+filename)
    
    # cv2.namedWindow("original")
    # cv2.moveWindow("original", 0, 30)
    # cv2.imshow("original", image)

    startY = 160
    endY = 325
    
    startX = 0
    endX = 848
    
    cropped = image[startY:endY, startX:endX]

    # cv2.namedWindow("cropped")
    # cv2.moveWindow("cropped", 920, 415)
    # cv2.imshow("cropped", cropped)

    concat = cv2.vconcat([image, cropped])
    cv2.namedWindow("Combined")
    cv2.moveWindow("Combined", int(width/2 - 424), int(height/2 - 300))
    cv2.imshow("Combined", concat)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cv2.destroyAllWindows()