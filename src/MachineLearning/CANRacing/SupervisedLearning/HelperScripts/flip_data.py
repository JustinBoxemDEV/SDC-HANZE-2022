# this seems to give really weird outputs do not use

import csv
import cv2
import os

#source path csv
sourcePathCSV = "D:\\2022 + flipped bochten 0.2\\2022_all_images.csv"

#source path images
sourcePathImages = "D:\\2022 + flipped bochten 0.2\\images 30-03-2022 15-22-30\\"

#target path flipped csv
targetPathCSV = "D:\\2022 + flipped bochten 0.2\\2022_full_flipped_csv\\flipped_images 30-03-2022 15-22-30.csv"

#target dir flipped images
targetPathImages = "D:\\2022 + flipped bochten 0.2\\2022_full_flipped\\flipped_images 30-03-2022 15-22-30\\"

readf = open(sourcePathCSV, "r")
writef = open(targetPathCSV, 'w')

fields=('Steer', 'Throttle', 'Brake', 'Image')

writer = csv.DictWriter(writef, fieldnames=fields, lineterminator='\n')
reader = csv.reader(readf)

header = next(reader)
writer.writeheader()

for row in reader:
    text = str(row).replace("[", "").replace("]", "").replace("'", "")
    data = text.split(",")
    steering = float(data[0])
    imagePath = str(data[3])
    # print(imagePath)
    imageData = imagePath.split("/")
    imageName = imageData[1]
    # print(imageName)

    for imageDir in os.listdir(sourcePathImages):
        if imageName == imageDir:
            image = cv2.imread(sourcePathImages+"\\"+imageDir)
            flippedImage = cv2.flip(image, 1)
            imageCsv = "flipped_images 30-03-2022 15-22-30"+"/"+"flipped_"+imageName
            if steering >= 0.2 or steering <= -0.2:
                steering = steering * -1
                cv2.imwrite(targetPathImages+"\\"+"flipped_"+imageName, flippedImage)
                writer.writerow({'Steer': steering, 'Throttle': data[1], 'Brake': data[2], 'Image': imageCsv})
                print(steering)
                # print("Added flipped steer image")

