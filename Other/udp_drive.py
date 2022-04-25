#/usr/bin/env python3

# ! ONLY COMPATIBLE WITH WINDOWS DUE TO WIN32GUI, NOT USED IN OUR PROJECT

import time
import struct
import os
import socket
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import winguiauto
import win32gui
from PIL import ImageGrab

CAN_MSG_SENDING_SPEED = .040 # 25Hz
IP = "127.0.0.1"
PORT = 5454

def main(model):
    #ACWindow = win32gui.FindWindow(None, "Assetto Corsa")
    capture = cv2.VideoCapture("training.avi")
    #capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    #capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    #capture.set(cv2.CAP_PROP_FOCUS, 0)
    #capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) #important to set right codec and enable the 60fps
    #capture.set(cv2.CAP_PROP_FPS, 60) #enable 60FPS

    # Performance gains
    ushort_to_bytes = struct.Struct('>H').pack
    float_to_bytes = struct.Struct('f').pack

    # Start running
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client_socket:
            while (True):
                start_time = time.time()
                #rect = win32gui.GetWindowPlacement(ACWindow)[-1]
                #frame = np.array(ImageGrab.grab(rect))[:,:,::-1]
                _, frame = capture.read()
                cv2.imshow("Een naam", frame)
                cv2.waitKey(1)
                
                frame = np.resize(np.array(cv2.resize(frame, image_size[::-1]))[:,:,::-1], (1, *image_size, 3)).astype(np.float32)
                steering_angle, throttle, brake = model.predict(frame)[0]
                
                client_socket.sendto(ushort_to_bytes(0x120) + bytes([round(throttle * 100)] + [8]*7), (IP, PORT))
                client_socket.sendto(ushort_to_bytes(0x126) + bytes([round(brake * 100)] + [0]*7), (IP, PORT))
                client_socket.sendto(ushort_to_bytes(0x12c) + float_to_bytes(steering_angle) + bytes([0]*4), (IP, PORT))

                end_time = time.time()
                time_diff = end_time - start_time
                if time_diff < CAN_MSG_SENDING_SPEED:
                    time.sleep(CAN_MSG_SENDING_SPEED-time_diff)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    model = load_model('drive_model.h5')
    image_size = (144,256)
    main(model)
