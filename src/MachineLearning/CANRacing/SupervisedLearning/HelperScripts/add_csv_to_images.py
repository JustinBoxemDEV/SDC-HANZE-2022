# Douwe script for splitting images with turns

import csv
import os

for imageDir in os.listdir("/home/douwe/Documents/recht"):
    images = []
    for csvFile in os.listdir("/home/douwe/Documents/all csv"):
        name = str(csvFile).replace("new_data images", "").replace(".csv", "")
        if(imageDir.__contains__(name)):
            for image in os.listdir("/home/douwe/Documents/recht/"+imageDir):
                images.append(image)
            readf = open(f"/home/douwe/Documents/all csv/"+csvFile, "r")
            reader = csv.reader(readf)
            with open("/home/douwe/Documents/python generated csv/"+imageDir+".csv", "w") as writef:
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
                print("Done " + imageDir + " : " + csvFile)