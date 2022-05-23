# this seems to give really weird outputs do not use

import csv
from os import listdir
import struct

readf = open('D:/SDC/sdc_data/justin_data/full_dataset/training/hanze_all_images.csv', "r")
writef = open('D:/SDC/sdc_data/justin_data/full_dataset/training/100_hanze_all_images.csv', 'w')


fields=('Steer', 'Throttle', 'Brake', 'Image')

writer = csv.DictWriter(writef, fieldnames=fields, lineterminator='\n')
reader = csv.reader(readf)

writer.writeheader()

for row in reader:
    text = str(row)

    data = text.split(",")
    # print("data:", data)

    throttle = data[2].replace(" ", "").replace("'", "")

    if (throttle == "0"):
        # print("uhoh")
        throttle == "100"

    # throttle = data[1].replace(" ", "").replace("'", "")

    if(len(throttle) == 0):
        throttle = 0
    elif(throttle == "100"):
        throttle = 100
    else:
        throttle = int(throttle) 
        throttle = throttle * 1.75438596491
        throttle = int(round(throttle))
        print(throttle)
        

    writer.writerow({'Steer': data[0], 'Throttle': throttle, 'Brake': data[2], 'Image': data[3]})
