""" Script for deplying a .pt model on the kart (Python version)

    This script will load a .pt model, extract a steering prediction from the model and then send it to the CANbus.
"""
import struct
import numpy as np
import cv2
import torch
from SelfDriveModel import SelfDriveModel
from transforms import Normalizer, ToTensor
import torchvision.transforms as transforms
import can

CAN_MSG_SENDING_SPEED = .040 # 25Hz

def main(model):
    # WEBCAM 
    capture = cv2.VideoCapture(0)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 848)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    capture.set(cv2.CAP_PROP_FOCUS, 0)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # important to set right codec and enable the 60fps
    capture.set(cv2.CAP_PROP_FPS, 60) # enable 60FPS

    bus = can.Bus(interface='socketcan', channel='vcan0', bitrate=500000)

    acc_msg = can.Message(arbitration_id=0x120, is_extended_id=True, data=[0, 0, 1, 0, 0, 0, 0, 0])
    acc_task = bus.send_periodic(acc_msg, CAN_MSG_SENDING_SPEED)
    
    steering_msg = can.Message(arbitration_id=0x12c, is_extended_id=True, data=[0, 0, 0, 0, 0, 0, 0, 0])
    steering_task = bus.send_periodic(steering_msg, CAN_MSG_SENDING_SPEED)

    # brake_msg = can.Message(arbitration_id=0x126, is_extended_id=True, data=[brake, 0, 0, 0, 0, 0, 0, 0])
    # brake_task = bus.send_periodic(brake_msg, CAN_MSG_SENDING_SPEED)

    # Start running
    while (True):
        _, frame = capture.read()

        frame = np.resize(frame, (848, 480, 3)).astype(np.float32)
        
        # transforms
        t = []
        t.append(Normalizer(0, 255))
        t.append(ToTensor())
        transform = transforms.Compose(t)
        frame = transform(frame)
        
        frame = frame.to(device=torch.device("cpu"))

        outputs = model(frame).detach().numpy()

        steering_angle, throttle, brake = outputs[0][0], int(outputs[0][1]), int(outputs[0][2])

        print("steer:", steering_angle, "throttle", int(throttle), "brake", int(brake))


        steering_msg.data = list(bytearray(struct.pack("f", float(steering_angle))) + bytearray(4))
        acc_msg.data = [throttle, 0, 1, 0, 0, 0, 0, 0]


        acc_task.modify_data(acc_msg)
        steering_task.modify_data(steering_msg)
        # brake_task.modify_data(brake_msg)



if __name__ == '__main__':
    device = torch.device("cpu")
    model = SelfDriveModel()
    model.load_state_dict(torch.load("./assets/models/final_seed1234_SteerSLSelfDriveModel_2022-06-05_13-13-52.pt", 
                            map_location=device))
    
    model.to("cpu")
    model.eval()
    main(model)