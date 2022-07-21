# Douwe script for splitting images with turns

import csv
import shutil
import os

# csvName = "new_data images 18-11-2021 14-59-21" # naam van de CSV (zonder extension)
# foldername = "good_images 18-11-2021 14-59-21" # naam van de folder met alle images

readf = open("D:\\Testing\\Testing\\testing_60p_100_new_data_images_30-03-2022_15-17-40_smoothed.csv", "r") # path naar de folder waar de csv in staat

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

    imageSource = f"D:\\Testing\\Testing\\images 30-03-2022 15-17-40\\{imagePath[27:]}" # path naar de locatie van de van de eerder gegeven folder (met alle images)
    # print(imageSource)
    # print(os.path.exists(imageSource))

    if os.path.exists(imageSource):
        imageDestinationbochten = "D:\\Testing\\Testing_bochten\\images 30-03-2022 15-17-40\\" # de locatie waar je de bochten op wil slaan
        imageDestinationBochtenRechts = "D:\\Testing\\Testing_bochten_rechts\\images 30-03-2022 15-17-40\\"
        imageDestinationBochtenLinks = "D:\\Testing\\Testing_bochten_links\\images 30-03-2022 15-17-40\\"
        imageDestinationrecht = "D:\\Testing\\Testing_recht\\images 30-03-2022 15-17-40\\" # de locatie waar je de rechte stukken op wil slaan

        if not os.path.exists(imageDestinationbochten):
            print(f"Creating bochten folder: {imageDestinationbochten}")
            os.mkdir(imageDestinationbochten) 

        if not os.path.exists(imageDestinationrecht):
            print(f"Creating recht folder: {imageDestinationrecht}")
            os.mkdir(imageDestinationrecht) 

        if((steering >= 0.25) or (steering <= -0.25)):
            print("copying to bochten!: ", imageDestinationbochten+imageName)
            shutil.copy(imageSource, imageDestinationbochten+imageName)
            print("bochten")
        # if((steering <= 0.015) and (steering >= -0.015)):
            # print("copying to recht!:", imageDestinationrecht+imageName)
            # shutil.copy(imageSource, imageDestinationrecht+imageName)
            #print("recht")
        if((steering >= 0.25)):
            shutil.copy(imageSource, imageDestinationBochtenRechts+imageName)
        if steering <= -0.25:
            shutil.copy(imageSource, imageDestinationBochtenLinks+imageName)

readf.close()
print("Done")