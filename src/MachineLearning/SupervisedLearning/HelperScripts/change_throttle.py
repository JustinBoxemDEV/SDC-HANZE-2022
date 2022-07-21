import os
import csv

throttle = 57

readf = open('C:\\Users\\douwe\\Desktop\\images 22-03-2022 13-26-11.csv', 'r')
reader = csv.reader(readf)

csvData = []

for row in reader:
    print(row)
    csvData.append(row)
    
readf.close()
print("Read done")

writef = open('C:\\Users\\douwe\\Desktop\\images 22-03-2022 13-26-11.csv', 'w')

fields=('Steer', 'Throttle', 'Brake', 'Image')
writer = csv.DictWriter(writef, fieldnames=fields, lineterminator='\n')

for line in csvData:
    data = str(row).replace("[", "").replace("]", "").replace("\'", "").replace(" ", "").split(",")
    writer.writerow({'Steer': data[0], 'Throttle': throttle, 'Brake': data[2], 'Image': data[3]})

writef.close()
print("Write done")