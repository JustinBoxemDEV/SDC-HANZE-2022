import csv
import os

for imageDir in os.listdir("D:\\SDC Data\\all images"):
    images = []
    for csvFile in os.listdir("D:\\SDC Data\\good csv"):
        name = str(csvFile).replace("new_data images", "").replace(".csv", "")
        if(imageDir.__contains__(name)):
            for image in os.listdir("D:\\SDC Data\\all images\\"+imageDir):
                images.append(image)
            readf = open("D:\\SDC Data\\good csv\\"+csvFile, "r")
            reader = csv.reader(readf)
            with open("D:\\SDC Data\\python generated csv\\"+imageDir+".csv", "w", newline="\n") as writef:
                header = next(reader)
                writer = csv.writer(writef)
                for row in reader:
                    data = str(row).split(",")
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