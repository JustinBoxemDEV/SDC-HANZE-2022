# rdw smoothing script

import os
from scipy.signal import savgol_filter
def main():
    #datafile = open("dataset_2022/training/images_old.csv", "r")
    #dataoutfile = open("dataset_2022/training/images.csv", "w")
    #datafile = open("dataset_2022/validation/val_old.csv", "r")
    #dataoutfile = open("dataset_2022/validation/val.csv", "w")
    datafile = open("dataset_2022/testing/test_old.csv", "r")
    dataoutfile = open("dataset_2022/testing/test.csv", "w")
    linebuffer = []
    steeringbuffer = []
    for line in datafile.readlines():
        x = line.split(",")
        steeringbuffer.append(float(x[0]))
        linebuffer.append(",".join(x[1:]))
    steeringbuffer = savgol_filter(steeringbuffer, 100, 3)
    for i in range(len(linebuffer)):
        dataoutfile.write(str(steeringbuffer[i]) + "," + linebuffer[i])
if __name__ == '__main__':
    main()