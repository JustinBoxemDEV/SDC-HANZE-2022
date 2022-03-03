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
    
    acc_msg = can.Message(arbitration_id=0x120, is_extended_id=True, data=[0, 0, 1, 0, 0, 0, 0, 0])
    acc_task = bus.send(acc_msg)

    time.sleep(0.10)
    acc_msg = can.Message(arbitration_id=0x126, is_extended_id=True, data=[0, 0, 0, 0, 0, 0, 0, 0])
    acc_task = bus.send(acc_msg)
    
    
    acc_msg = can.Message(arbitration_id=0x120, is_extended_id=True, data=[0, 0, 1, 0, 0, 0, 0, 0])
    acc_task = bus.send_periodic(acc_msg, CAN_MSG_SENDING_SPEED)
    
    direction = 0
    try:
        while (True):
            changeDir = input("Change direction? (F/R/N) > ").lower()
            #changeDir = "n"
            if "f" in changeDir:
            	direction = 1
            	acc_msg = can.Message(arbitration_id=0x120, is_extended_id=True, data=[10, 0, 1, 0, 0, 0, 0, 0])
            	acc_task.modify_data(acc_msg)
            if "r" in changeDir:
            	direction = 2
            	acc_msg = can.Message(arbitration_id=0x120, is_extended_id=True, data=[10, 0, 2, 0, 0, 0, 0, 0])
            	acc_task.modify_data(acc_msg)
            if "n" in changeDir:
            	direction = 0
            	acc_msg = can.Message(arbitration_id=0x120, is_extended_id=True, data=[10, 0, 0, 0, 0, 0, 0, 0])
            	acc_task.modify_data(acc_msg)
                
                
            try:
                newSpeed = int(input("Speed > "))
                print(direction)
                acc_msg = can.Message(arbitration_id=0x120, is_extended_id=True, data=[newSpeed, 0, direction, 0, 0, 0, 0, 0])
                print(acc_msg)
                acc_task.modify_data(acc_msg)
            except:
                print("Invalid command")
            
    except KeyboardInterrupt:
        pass
    acc_task.stop()
    

if __name__ == '__main__':
    main()

