import csv
import struct
import cv2

readf = open('/home/douwe/Desktop/data/data_images_18-11-2021_15-12-21.csv')
writef = open('/home/douwe/Desktop/data/test.csv', 'w')

fields=('Steer', 'Throttle', 'Brake', 'Image')

writer = csv.DictWriter(writef, fieldnames=fields, lineterminator='\n')
reader = csv.reader(readf)

writer.writeheader()
header = next(reader)

for row in reader:
    text = str(row)
    text = text.replace("\"", "").replace("[", "").replace("]", "").replace("\'", "")

    # print(text)
    # text = text.replace(";", "")

    data = text.split("|")

    image = data[4].replace("\"", "")
    steering = data[0].replace("\"", "").split(",")
    steering.reverse()
    steeringData = [int(x) for x in steering]
    steeringData.reverse()
    steeringBytes = bytes(steeringData)

    steerFloat = struct.unpack('f', steeringBytes[0:4])
    steerFloat = str(steerFloat).replace("(", "").replace(")", "").replace(",", "")
    steerFloat = float(steerFloat)

    imageFloat = image.replace(".png", ".jpg").split("/")
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

writef.close()
readf.close()