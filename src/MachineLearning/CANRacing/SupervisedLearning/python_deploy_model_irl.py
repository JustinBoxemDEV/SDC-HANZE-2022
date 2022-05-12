#/usr/bin/env python3

# ! ONLY COMPATIBLE WITH WINDOWS DUE TO WIN32GUI, NOT USED IN OUR PROJECT

import time
import struct
import os
import socket
import numpy as np
import cv2
import torch
from PIL import ImageGrab
from SelfDriveModel import SelfDriveModel
from transforms import Normalizer, ToTensor
import torchvision.transforms as transforms
import skimage.io
import matplotlib.pyplot as plt
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

    # Start running
    while (True):
        start_time = time.time()

        _, frame = capture.read()

        frame = np.resize(frame, (480, 848, 3)).astype(np.float32)
        
        # transforms
        t = []
        t.append(Normalizer(0, 255))
        t.append(ToTensor())
        transform = transforms.Compose(t)
        frame = transform(frame)
        
        frame = frame.to(device=torch.device("cpu"))

        outputs = model(frame).detach().numpy()

        steering_angle, throttle, brake = outputs[0][0], outputs[0][1], outputs[0][2]

        print("steer:", steering_angle, "throttle", int(throttle), "brake", int(brake))

        
        # TODO: fix sending messages to CAN :(
        bus = can.Bus(interface='socketcan', channel='can0', bitrate=500000)

        acc_msg = can.Message(arbitration_id=0x120, is_extended_id=True, data=[throttle, 0, 1, 0, 0, 0, 0, 0])
        acc_task = bus.send_periodic(acc_msg, CAN_MSG_SENDING_SPEED)
        
        steering_msg = can.Message(arbitration_id=0x12c, is_extended_id=True, data=[0, 0, 0, 0, 0, 0, 0, 0])
        steering_msg.data = list(bytearray(struct.pack("f", float(steering_angle))) + bytearray(4))
        steering_task = bus.send_periodic(steering_msg, CAN_MSG_SENDING_SPEED)

        brake_msg = can.Message(arbitration_id=0x126, is_extended_id=True, data=[brake, 0, 0, 0, 0, 0, 0, 0])
        brake_task = bus.send_periodic(brake_msg, CAN_MSG_SENDING_SPEED)

        acc_task.start()
        steering_task.start()
        brake_task.start()

        end_time = time.time()
        time_diff = end_time - start_time
        if time_diff < CAN_MSG_SENDING_SPEED:
            time.sleep(CAN_MSG_SENDING_SPEED-time_diff)


if __name__ == '__main__':
    device = torch.device("cpu")
    model = SelfDriveModel()
    model.load_state_dict(torch.load("src/MachineLearning/CANRacing/models/SLSelfDriveModel.pt", 
                            map_location=device))
    
    model.to("cpu")
    model.eval()
    main(model)