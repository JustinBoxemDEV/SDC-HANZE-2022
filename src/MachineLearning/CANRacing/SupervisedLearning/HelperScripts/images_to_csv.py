import os
import csv

readf = open("D:\\Testing\\Testing\\testing_60p_100_new_data_images_30-03-2022_15-17-40_smoothed.csv", 'r')
reader = csv.reader(readf)
header = next(reader)

writef = open("D:\\Testing\\Testing_recht\\Testing_recht.csv", 'w')

fields=('Steer', 'Throttle', 'Brake', 'Image')

writer = csv.DictWriter(writef, fieldnames=fields, lineterminator='\n')
reader = csv.reader(readf)

writer.writeheader()

images = []

for image in os.listdir("D:\\Testing\\Testing_recht\\images 30-03-2022 15-17-40\\"):
    images.append(image)
    

index = 0

for row in reader:
    for image in images:
        index = index + 1
        steering = float(row[0])
        throttle = int(row[1])
        brake = int(row[2])
        imageData = str(row[3])
        imageList = imageData.split("/")
        imagePath = imageList[0]
        imageName = imageList[1]

        # print("row image: "+imageName)
        # print("image: " + image)
        # print(imageName)
        if str(image) == imageName:
            writer.writerow({'Steer': steering, 'Throttle': throttle, 'Brake': brake, 'Image': imageData})
            images.remove(image)