import csv
import struct
import cv2
import os
from os import listdir

readf = open('/home/douwe/Desktop/data/data_images_18-11-2021_15-12-21.csv')
writef = open('/home/douwe/Desktop/data/test.csv', 'w')

twentytwo = False
twentyone = False
validation = True

fields=('Steer', 'Throttle', 'Brake', 'Image')

writer = csv.DictWriter(writef, fieldnames=fields, lineterminator='\n')
reader = csv.reader(readf)

writer.writeheader()
header = next(reader)

if(validation):
    writevalidation = open('/home/douwe/Desktop/data/test2.csv', 'w')
    writer = csv.DictWriter(writevalidation, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    
    imageArray = []
    images = '/home/douwe/Desktop/data/bochten/'
    for imageFolder in os.listdir(images):
        imageArray.append(imageFolder)
    
    for row in reader:
        for imageFolder in imageArray:
            text = str(row)
            text = text.replace("\"", "").replace("[", "").replace("]", "").replace("\'", "")

            # print(text)
            # text = text.replace(";", "")

            data = text.split("|")

            image = data[4].replace("\"", "").replace(".png", ".jpg")

            imageFloat = image.split("/")
            
            if(imageFloat[1] == imageFolder):
                imageArray.remove(imageFolder)
                steering = data[0].replace("\"", "").split(",")
                steering.reverse()
                steeringData = [int(x) for x in steering]
                steeringData.reverse()
                steeringBytes = bytes(steeringData)

                steerFloat = struct.unpack('f', steeringBytes[0:4])
                steerFloat = str(steerFloat).replace("(", "").replace(")", "").replace(",", "")
                steerFloat = float(steerFloat)
                print("True")

                # print(imageFloat)
                # imageName = imageFloat[1]
                # print(imageName)
                
                # imageSource = '/home/douwe/Desktop/data/'+imageFloat[0]+'/'+imageFloat[1]
                
                # rdwimage = cv2.imread(imageSource)
                
                # if((steerFloat > 0.1) or (steerFloat < -0.1)):  
                #     imageDestination = '/home/douwe/Desktop/data/bochten/'+imageName
                #     cv2.imwrite(imageDestination, rdwimage)
                # if(steerFloat == 0):
                #     imageDestination = '/home/douwe/Desktop/data/rechte stukken/'+imageName
                #     cv2.imwrite(imageDestination, rdwimage)

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
                writer.writerow({'Steer': steerFloat, 'Throttle': throttle, 'Brake': brake, 'Image': image})
    # writer.writeheader()    
    
    
    
    # for row in reader:
    #     text = str(row)
    #     text = text.replace("\"", "").replace("[", "").replace("]", "").replace("\'", "")

    #     # print(text)
    #     # text = text.replace(";", "")

    #     data = text.split("|")

    #     image = data[4].replace("\"", "").replace(".png", ".jpg")
    #     steering = data[0].replace("\"", "").split(",")
    #     steering.reverse()
    #     steeringData = [int(x) for x in steering]
    #     steeringData.reverse()
    #     steeringBytes = bytes(steeringData)

    #     steerFloat = struct.unpack('f', steeringBytes[0:4])
    #     steerFloat = str(steerFloat).replace("(", "").replace(")", "").replace(",", "")
    #     steerFloat = float(steerFloat)

    #     imageFloat = image.split("/")
    #     # print(imageFloat)
    #     imageName = imageFloat[1]
    #     # print(imageName)
        
    #     imageSource = '/home/douwe/Desktop/data/'+imageFloat[0]+'/'+imageFloat[1]
        
    #     rdwimage = cv2.imread(imageSource)
        
    #     if((steerFloat > 0.1) or (steerFloat < -0.1)):  
    #         imageDestination = '/home/douwe/Desktop/data/bochten/'+imageName
    #         cv2.imwrite(imageDestination, rdwimage)
    #     if(steerFloat == 0):
    #         imageDestination = '/home/douwe/Desktop/data/rechte stukken/'+imageName
    #         cv2.imwrite(imageDestination, rdwimage)

    #     throttle = data[1].replace("\"", "").split(",")[0]
    #     if(len(throttle) == 0):
    #         throttle = 0
    #     else:
    #         throttle = int(throttle) 
    #         throttle = throttle * 1.75438596491
    #         throttle = int(round(throttle))
        
    #     brake = data[2].replace("\"", "").split(",")[0]
    #     if(len(brake) == 0):
    #         brake = 0
    #     else:
    #         brake = int(brake)
    #     writer.writerow({'Steer': steerFloat, 'Throttle': throttle, 'Brake': brake, 'Image': image})

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
        
        imageSource = '/home/douwe/Desktop/data/'+imageFloat[0]+'/'+imageFloat[1]
        
        rdwimage = cv2.imread(imageSource)
        
        if((steerFloat > 0.1) or (steerFloat < -0.1)):  
            imageDestination = '/home/douwe/Desktop/data/bochten/'+imageName
            cv2.imwrite(imageDestination, rdwimage)
        if(steerFloat == 0):
            imageDestination = '/home/douwe/Desktop/data/rechte stukken/'+imageName
            cv2.imwrite(imageDestination, rdwimage)

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
        
        imageSource = '/home/douwe/Desktop/data/'+imageFloat[0]+'/'+imageFloat[1]
        
        rdwimage = cv2.imread(imageSource)
        
        if((steerFloat > 0.1) or (steerFloat < -0.1)):  
            imageDestination = '/home/douwe/Desktop/data/bochten/'+imageName
            cv2.imwrite(imageDestination, rdwimage)
        if(steerFloat == 0):
            imageDestination = '/home/douwe/Desktop/data/rechte stukken/'+imageName
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