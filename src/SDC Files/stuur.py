#/usr/bin/env python3

import numpy as np
import time
import can
import struct
import os


os.system("sudo ip link set can0 type can bitrate 500000")
os.system("sudo ip link set can0 up")

CAN_MSG_SENDING_SPEED = .04 # 100Hz



def main():
    bus = can.Bus(interface='socketcan', channel='can0', bitrate=500000)
    
    # home steering wheel
    homing_msg = can.Message(arbitration_id=0x6f1, data=[0, 0, 0, 0, 0, 0, 0, 0])
    bus.send(homing_msg)
    
    # set up periodic steering message
    steering_msg = can.Message(arbitration_id=0x12c, data=[0, 0, 0, 0, 0, 0, 0, 0])
    steering_task = bus.send_periodic(steering_msg, CAN_MSG_SENDING_SPEED)
    
    direction = 1
    try:
        while (True):
            newAngle = float(input("Steering angle (-1.0 | 1.0) > "))
            steering_msg = can.Message(arbitration_id=0x12c, data=list(bytearray(struct.pack("f", newAngle)) + bytearray(4)))
            steering_task.modify_data(steering_msg)

            
    except KeyboardInterrupt:
        pass
    steering_task.stop()
    

if __name__ == '__main__':
    main()



