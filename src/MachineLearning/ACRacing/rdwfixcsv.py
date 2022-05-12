import csv
import struct

readf = open('C:\\Users\\douwe\\Desktop\\data images 18-11-2021 11-07-14.csv')
writef = open('C:\\Users\\douwe\Desktop\\test.csv', 'w')

fields=('Steer', 'Throttle', 'Brake', 'Image')

writer = csv.DictWriter(writef, fieldnames=fields, lineterminator='\n')
reader = csv.reader(readf)

writer.writeheader()
header = next(reader)

for row in reader:
    print(row)

    text = str(row)
    text = text.replace("\"", "").replace("[", "").replace("]", "").replace("\'", "")
    print(text)

    # print(text)
    # text = text.replace(";", "")

    data = text.split("|")

    image = data[4].replace("\"", "")
    steering = data[0].replace("\"", "").split(",")
    steering.reverse()
    steeringData = [int(x) for x in steering]
    steeringData.reverse()
    steeringBytes = bytes(steeringData)

    print(steeringBytes)

    steerFloat = struct.unpack('f', steeringBytes[0:4])
    steerFloat = str(steerFloat).replace("(", "").replace(")", "").replace(",", "")
    steerFloat = float(steerFloat)

    throttle = data[1].replace("\"", "").split(",")[0]
    if(len(throttle) == 0):
        throttle = 0
    else:
       throttle = int(throttle) 
       throttle = throttle * 1.75438596491
       throttle = int(round(throttle))
       print(throttle)
    
    brake = data[2].replace("\"", "").split(",")[0]
    if(len(brake) == 0):
        brake = 0
    else:
        brake = int(brake)
    writer.writerow({'Steer': steerFloat, 'Throttle': throttle, 'Brake': brake, 'Image': image})

writef.close()
readf.close()