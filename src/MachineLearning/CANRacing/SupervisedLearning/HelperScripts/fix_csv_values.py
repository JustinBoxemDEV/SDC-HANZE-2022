# Douwe script for fixing csv values
import csv
import struct

csvName = "data images 22-03-2022 13-47-20"

readf = open(f"C:/Users/Sabin/Documents/SDC/SL_data/zips/{csvName}.csv", "r")
writef = open(f"C:/Users/Sabin/Documents/SDC/SL_data/zips/new_{csvName}.csv", 'w')

twentytwo = True
twentyone = False 

fields=('Steer', 'Throttle', 'Brake', 'Image')

writer = csv.DictWriter(writef, fieldnames=fields, lineterminator='\n')
reader = csv.reader(readf)

writer.writeheader()
header = next(reader)

if(twentyone):
    for row in reader:
        text = str(row)
        text = text.replace("\"", "").replace("[", "").replace("]", "").replace("\'", "")

        data = text.split("|")

        image = data[4].replace("\"", "").replace(".png", ".jpg")
        steering = data[0].replace("\"", "").split(",")
        steering.reverse()
        steeringData = [int(x) for x in steering]
        steeringData.reverse()
        steeringBytes = bytes(steeringData)

        steerFloat = struct.unpack('f', steeringBytes[0:4])
        steerFloat = str(steerFloat).replace("(", "").replace(")", "").replace(",", "")
        steerFloat = float(steerFloat)

        imageFloat = image.split("/")
        imageName = imageFloat[1]

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

if(twentytwo):
    for row in reader:
        text = str(row)
        text = text.replace("\"", "").replace("[", "").replace("]", "").replace("\'", "")

        data = text.split("|")

        image = data[3].replace("\"", "")
        steering = float(data[0])

        imageFloat = image.split("/")

        imageName = imageFloat[1]

        throttle = data[1].replace("\"", "").split(",")[0]
        if(len(throttle) == 0):
            throttle = 0
        else:
            throttle = float(throttle) 
            throttle = int(throttle * 100)
            throttle = int(round(throttle))
        
        brake = float(data[2])
        brake = int(brake * 100)
    
        writer.writerow({'Steer': steering, 'Throttle': throttle, 'Brake': brake, 'Image': image})

writef.close()
readf.close()