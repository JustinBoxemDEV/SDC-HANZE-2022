# Douwe script for splitting images with turns

import csv
import shutil
import cv2
import os
from os import listdir

readf = open('C:/Users/Sabin/Documents/SDC/RDW_Data/new/good_data images 18-11-2021 12-53-47.csv', "r")
writef = open('C:/Users/Sabin/Documents/SDC/RDW_Data/bochten/douwe_good_data images 18-11-2021 12-53-47.csv', 'w')

fields=('Steer', 'Throttle', 'Brake', 'Image')

writer = csv.DictWriter(writef, fieldnames=fields, lineterminator='\n')
reader = csv.reader(readf)

writer.writeheader()
header = next(reader)
  
for row in reader:
    text = str(row)
    data = text.split(",")

    steering = float(data[0])
    throttle = int(data[1])
    brake = int(data[2])
    imagePath = data[3]

    imageDirectory = imagePath.split("/")[0]
    imageName = imagePath.split("/")[1]

    # good_images 18-11-2021 12-53-47/
    imageSource = 'C:/Users/Sabin/Documents/SDC/RDW_Data/new/'+imagePath

    if os.path.exists(imageSource):
        rdwimage = cv2.imread(imageSource)
        
        if((steering > 0.1) or (steering < -0.1)):  
            imageDestination = 'C:/Users/Sabin/Documents/SDC/RDW_Data/bochten/good_images 18-11-2021 12-53-47/'+imageName
            shutil.copy(imageSource, imageDestination)
        if(steering == 0):
            imageDestination = 'C:/Users/Sabin/Documents/SDC/RDW_Data/recht/good_images 18-11-2021 12-53-47/'+imageName
            shutil.copy(imageSource, imageDestination)

        writer.writerow({'Steer': steering, 'Throttle': throttle, 'Brake': brake, 'Image': imagePath})

writef.close()
readf.close()