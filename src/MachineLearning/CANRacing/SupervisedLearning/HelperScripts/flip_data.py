# this seems to give really weird outputs do not use

import csv
import cv2
import os

#source path csv
sourcePathCSV = "C:\\Users\\douwe\\Desktop\\test_dataset\\test_csv.csv"

#source path images
sourcePathImages = "C:\\Users\\douwe\\Desktop\\test_dataset\\test_images"

#target path flipped csv
targetPathCSV = "C:\\Users\\douwe\\Desktop\\flipped data\\Flipped CSV\\test_csv.csv"

#target dir flipped images
targetPathImages = "C:\\Users\\douwe\\Desktop\\flipped data\\Flipped Images"

readf = open(sourcePathCSV, "r")
writef = open(targetPathCSV, 'w')

fields=('Steer', 'Throttle', 'Brake', 'Image')

writer = csv.DictWriter(writef, fieldnames=fields, lineterminator='\n')
reader = csv.reader(readf)

header = next(reader)
writer.writeheader()

for imageName in os.listdir(sourcePathImages):
    image = cv2.imread(sourcePathImages+"\\"+imageName)
    flippedImage = cv2.flip(image, 1)
    cv2.imwrite(targetPathImages+"\\"+imageName, flippedImage)

for row in reader:
    text = str(row).replace("[", "").replace("]", "").replace("'", "")
    data = text.split(",")
    steering = float(data[0])

    if(steering != 0):
        steering = steering * -1

    print(steering)
    writer.writerow({'Steer': steering, 'Throttle': data[1], 'Brake': data[2], 'Image': data[3]})
