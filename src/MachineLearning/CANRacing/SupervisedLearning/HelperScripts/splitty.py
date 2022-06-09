# split datasets in left right straight for classification
import os
f = open("/new_data images 07-06-2022 16-37-30.csv", "r", encoding='utf-8-sig')
for line in f.readlines():
    data = line.split(",")
    steering = float(data[0])
    filename = data[-1][:-1]
    # print(filename)
    # print(os.listdir())
    if steering > 0.1: os.rename(filename, "right/" + filename.split("/")[-1])
    elif steering < -0.1: os.rename(filename, "left/" + filename.split("/")[-1])
    elif steering < 0.1 or steering > -0.1: os.rename(filename, "straight/" + filename.split("/")[-1])