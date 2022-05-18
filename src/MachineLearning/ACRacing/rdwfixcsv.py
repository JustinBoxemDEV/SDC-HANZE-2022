import csv
import struct
import cv2
import os

from sympy import Q

readf = open("D:\\SL_data\\training\\data images 18-11-2021 15-12-21.csv")
writef = open('C:\\Users\\douwe\Desktop\\test.csv', 'w')

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
    # print(image)

    steering = data[0].replace("\"", "").split(",")
    steering.reverse()
    steeringData = [int(x) for x in steering]
    steeringData.reverse()
    steeringBytes = bytes(steeringData)

    steerFloat = struct.unpack('f', steeringBytes[0:4])
    steerFloat = str(steerFloat).replace("(", "").replace(")", "").replace(",", "")
    steerFloat = float(steerFloat)

    if steerFloat == 0.0:
        imageName = image.replace(".png", ".jpg").split("/")
        turnImage = cv2.imread("D:\\SL_data\\training\\"+imageName[0]+"\\"+imageName[1])
        turnImage = cv2.resize(turnImage, (640, 480))
        path = "D:\\SL_data\\training\\recht stukken 18-11-2021 15-12-21"
        cv2.imwrite(os.path.join(path, imageName[1]), turnImage)

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