# Douwe script for splitting images with turns

import csv
import os
  
fields=('Data')
  
for bochtDir in os.listdir("/home/douwe/Documents/bochten"):
    images = []
    directoryName = str(bochtDir)
    for csvFile in os.listdir("/home/douwe/Documents/all csv"):
        name = str(csvFile).replace("new_data images", "").replace(".csv", "")
        if(bochtDir.__contains__(name)):
            for image in os.listdir("/home/douwe/Documents/bochten/"+bochtDir):
                images.append(image)
            readf = open(f"/home/douwe/Documents/all csv/"+csvFile, "r")
            reader = csv.reader(readf)
            with open("/home/douwe/Documents/python generated csv/"+csvFile, "w") as writef:
                header = next(reader)
                writer = csv.writer(writef)
                
                for row in reader:
                    text = row
                    data = str(text).split(",")
                    imagePath = data[3].replace("\"", "").replace("[", "").replace("]", "").replace("\'", "")[1:]
                    
                    imageDirectory = imagePath.split("/")[0]
                    imageName = imagePath.split("/")[1]
                    for image in images:                
                        if(imageName == image):
                            writer.writerow(row)
                            images.remove(imageName)

                readf.close()
                writef.close()
                print("Done " + bochtDir + " : " + csvFile)