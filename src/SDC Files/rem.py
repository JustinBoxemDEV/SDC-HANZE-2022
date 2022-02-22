#/usr/bin/env python3

import numpy as np
import time
import can
import struct
import os


os.system("sudo ip link set can0 type can bitrate 500000")
os.system("sudo ip link set can0 up")
print()

CAN_MSG_SENDING_SPEED = .04 # 100Hz



def main():
    bus = can.Bus(interface='socketcan', channel='can0', bitrate=500000)
    
    acc_msg = can.Message(arbitration_id=0x126, is_extended_id=True, data=[0, 0, 0, 0, 0, 0, 0, 0])
    acc_task = bus.send_periodic(acc_msg, CAN_MSG_SENDING_SPEED)
    
    
    try:
        while (True):
            brake = int(input("Brake % > "))
            acc_msg = can.Message(arbitration_id=0x126, is_extended_id=True, data=[brake, 0, 0, 0, 0, 0, 0, 0])
            acc_task.modify_data(acc_msg)
            
    except KeyboardInterrupt:
        pass
    acc_task.stop()
    

if __name__ == '__main__':
    main()



