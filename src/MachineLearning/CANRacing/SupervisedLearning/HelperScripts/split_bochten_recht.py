# Douwe script for splitting images with turns

import csv
import shutil
import os

csvName = "new_data images 18-11-2021 14-59-21"
foldername = "good_images 18-11-2021 14-59-21"

readf = open(f"D:/SDC/sdc_data/RDW_Data/new/{csvName}.csv", "r")

fields=('Steer', 'Throttle', 'Brake', 'Image')

reader = csv.reader(readf)

header = next(reader)
  
for row in reader:
    text = str(row)
    data = text.split(",")
    data[0] = data[0].replace("\"", "").replace("[", "").replace("]", "").replace("\'", "").replace(" ", "")
    data[1] = data[1].replace("\"", "").replace("[", "").replace("]", "").replace("\'", "").replace(" ", "")
    data[2] = data[2].replace("\"", "").replace("[", "").replace("]", "").replace("\'", "").replace(" ", "")
    
    steering = float(data[0])
    throttle = int(data[1])
    brake = int(data[2])
    imagePath = data[3].replace("\"", "").replace("[", "").replace("]", "").replace("\'", "")[1:]

    imageDirectory = imagePath.split("/")[0]
    imageName = imagePath.split("/")[1]

    imageSource = f"D:/SDC/sdc_data/RDW_Data/new/{foldername}/{imagePath[27:]}"
    # print(imageSource)
    # print(os.path.exists(imageSource))

    if os.path.exists(imageSource):
        imageDestinationbochten = f"D:/SDC/sdc_data/RDW_Data/bochten/{foldername}/"
        imageDestinationrecht = f"D:/SDC/sdc_data/RDW_Data/recht/{foldername}/"

        if not os.path.exists(imageDestinationbochten):
            print(f"Creating bochten folder: {imageDestinationbochten}")
            os.mkdir(f"D:/SDC/sdc_data/RDW_Data/bochten/{foldername}")

        if not os.path.exists(imageDestinationrecht):
            print(f"Creating recht folder: {imageDestinationrecht}")
            os.mkdir(f"D:/SDC/sdc_data/RDW_Data/recht/{foldername}/")

        if((steering > 0.1) or (steering < -0.1)):
            # print("copying to bochten!: ", imageDestinationbochten+imageName)
            shutil.copy(imageSource, imageDestinationbochten+imageName)
        if(steering == 0):
            # print("copying to recht!:", imageDestinationrecht+imageName)
            shutil.copy(imageSource, imageDestinationrecht+imageName)

readf.close()
print("Done")