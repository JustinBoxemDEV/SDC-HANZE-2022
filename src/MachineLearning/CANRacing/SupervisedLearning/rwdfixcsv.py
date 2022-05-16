# Douwe script for splitting images with turns

import csv
import shutil
import struct
import cv2
import os
from os import listdir

readf = open('C:/Users/Sabin/Documents/SDC/RDW_Data/new/good_data images 18-11-2021 12-53-47.csv', "r")
writef = open('C:/Users/Sabin/Documents/SDC/RDW_Data/bochten/douwe_good_data images 18-11-2021 12-53-47.csv', 'w')

twentytwo = False
twentyone = True 

fields=('Steer', 'Throttle', 'Brake', 'Image')

writer = csv.DictWriter(writef, fieldnames=fields, lineterminator='\n')
reader = csv.reader(readf)

writer.writeheader()
header = next(reader)
  
if(twentyone):
    for row in reader:
        text = str(row)
        text = text.replace("\"", "").replace("[", "").replace("]", "").replace("\'", "")

        # print(text)
        # text = text.replace(";", "")

        data = text.split("|")

        image = data[4].replace("\"", "").replace(".png", ".jpg")
        steering = data[0].replace("\"", "").split(",")
        steering.reverse()
        steeringData = [int(x) for x in steering]
        steeringData.reverse()
        steeringBytes = bytes(steeringData)

        steerFloat = struct.unpack('f', steeringBytes[0:4])
        steerFloat = str(steerFloat).replace("(", "").replace(")", "").replace(",", "")
        steerFloat = float(steerFloat)

        imageFloat = image.split("/")
        # print(imageFloat)
        imageName = imageFloat[1]
        # print(imageName)
        # print(imageFloat)
        
        # good_images 18-11-2021 12-53-47/
        imageSource = 'C:/Users/Sabin/Documents/SDC/RDW_Data/new/'+imageFloat[0]+'/'+imageFloat[1]
        print(imageSource)
        if os.path.exists(imageSource):
            rdwimage = cv2.imread(imageSource)
            
            if((steerFloat > 0.1) or (steerFloat < -0.1)):  
                imageDestination = 'C:/Users/Sabin/Documents/SDC/RDW_Data/bochten/good_images 18-11-2021 12-53-47/'+imageName
                shutil.copy(imageSource, imageDestination)
            if(steerFloat == 0):
                imageDestination = 'C:/Users/Sabin/Documents/SDC/RDW_Data/recht/good_images 18-11-2021 12-53-47/'+imageName
                shutil.copy(imageSource, imageDestination)

            throttle = data[1].replace("\"", "").split(",")[0]
            if(len(throttle) == 0):
                throttle = 0
            else:
                throttle = int(throttle) 
                throttle = throttle * 1.75438596491
                throttle = int(round(throttle))
            
            brake = data[2].replace("\"", "").split(",")[0]
            if(len(brake) == 0):
                brake = 0
            else:
                brake = int(brake)
            
            if((steerFloat > 0.1) or (steerFloat < -0.1)): 
                writer.writerow({'Steer': steerFloat, 'Throttle': throttle, 'Brake': brake, 'Image': image})

if(twentytwo):
    for row in reader:
        text = str(row)
        text = text.replace("\"", "").replace("[", "").replace("]", "").replace("\'", "")

        # print(text)
        # text = text.replace(";", "")

        data = text.split("|")

        image = data[3].replace("\"", "")
        steering = float(data[0])

        imageFloat = image.split("/")
        # print(imageFloat)
        imageName = imageFloat[1]
        # print(imageName)
        
        imageSource = 'C:/Users/Sabin/Documents/SDC/RDW_Data/new'+imageFloat[0]+'/'+imageFloat[1]
        
        rdwimage = cv2.imread(imageSource)
        
        if((steerFloat > 0.1) or (steerFloat < -0.1)):  
            imageDestination = 'C:/Users/Sabin/Documents/SDC/RDW_Data/new/bochten/'+imageName
            cv2.imwrite(imageDestination, rdwimage)
        if(steerFloat == 0):
            imageDestination = 'C:/Users/Sabin/Documents/SDC/RDW_Data/new/rechte_stukken/'+imageName
            cv2.imwrite(imageDestination, rdwimage)

        throttle = data[1].replace("\"", "").split(",")[0]
        if(len(throttle) == 0):
            throttle = 0
        else:
            throttle = float(throttle) 
            throttle = int(throttle * 100)
            throttle = int(round(throttle))
        
        brake = float(data[2])
        print("brake: ")
        print(brake)
        brake = int(brake * 100)
    
        writer.writerow({'Steer': steering, 'Throttle': throttle, 'Brake': brake, 'Image': image})

writef.close()
readf.close()