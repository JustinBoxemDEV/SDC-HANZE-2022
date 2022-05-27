import os
def main():
    zerocutoff = 15
    minsteering = 10
    minsteeringvalue = 0.1
    datafile = open("dataset_2022/validation/val_old.csv", "r")
    dataoutfile = open("dataset_2022/validation/val.csv", "w")
    linebuffer = []
    steerbuffer = []
    zerocount = 0
    for line in datafile.readlines():
        x = line.split(",")
        if (x[0] == "0.0" or abs(float(x[0])) < minsteeringvalue) and steerbuffer == []:
            dataoutfile.write("0.0," + ",".join(x[1:]))
        elif x[0] == "0.0" or abs(float(x[0])) < minsteeringvalue:
            zerocount += 1
            linebuffer.append(",".join(x[1:]))
            steerbuffer.append(float(0))
            if zerocount == zerocutoff:
                zerocount = 0
                end = linebuffer[-3:]
                linebuffer = linebuffer[:-3]
                steerbuffer = steerbuffer[:-3]
                steering = 0
                if len(steerbuffer) > minsteering: steering = sum(steerbuffer) / len(steerbuffer)
                for item in linebuffer:
                    dataoutfile.write(str(steering) + "," + item)
                for item in end:
                    dataoutfile.write("0.0," + item)
                linebuffer, steerbuffer = [], []
        else:
            linebuffer.append(",".join(x[1:]))
            steerbuffer.append(float(x[0]))
            zerocount = 0
        #steer = float(x[0])
    if len(steerbuffer) > minsteering: steering = sum(steerbuffer) / len(steerbuffer)
    for item in linebuffer:
        dataoutfile.write(str(steering) + "," + item)
if __name__ == '__main__':
    main()