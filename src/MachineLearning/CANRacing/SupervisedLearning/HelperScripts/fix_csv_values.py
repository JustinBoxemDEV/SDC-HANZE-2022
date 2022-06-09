# Douwe script for fixing csv values
import csv
import struct

readf = open("D:/data images 07-06-2022 16-37-30.csv", "r") 
writef = open("D:/new_data images 07-06-2022 16-37-30.csv", 'w')

twentytwo = False
twentyone = True

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
            pass
            # throttle = int(throttle) 
            # throttle = throttle * 1.75438596491
            # throttle = int(round(throttle))
        
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
            throttle = int(throttle)

        brake = data[2].replace("\"", "").split(",")[0]
        
        if(len(throttle)==0):
            brake = 0
        else:
            brake = int(brake)
    
        print(steering, "," + throttle + "," + brake + "," + image)

        writer.writerow({'Steer': steering, 'Throttle': throttle, 'Brake': brake, 'Image': image})

writef.close()
readf.close()