""" Python script for deplying a combination of a classification model with a steering prediction model.
    Predictions are extracted and sent to the CANBus
"""
import os
import cv2
import numpy as np
import struct
import can
import torch
from SelfDriveModel import SelfDriveModel
from DirectionClassificationModel import DirectionClassificationModel
from transforms import ToTensor
import torchvision.transforms as transforms
import torch.nn.functional as F

CAN_MSG_SENDING_SPEED = .04 # 100Hz
SUDO_PASSWORD = '' # ENTER PASSWORD HERE!

camera = True
preview = True
CAN = "can0" # if testing on video, change to ""
acc_speed = 60
straight_cutoff = 0.1 # for swerving
corner_cutoff = 0.28 # for steering too much/little
cornering_multiplier = 1.50

model_name = "assets/models/seed4_368-207_SteerSLSelfDriveModel_2022-06-07_00-03-49.pt"

classification_model = DirectionClassificationModel(gpu=False)
classification_model.load_state_dict(torch.load(f"assets/models/classification/best.pt", map_location=torch.device('cpu')))

frame_sizes = {
    "camera": (1280, 720),
    "model": (368, 207), # 848, 480
    "classification": (128, 128)
    }

def main(model, classification_model, frame_sizes, acc_speed, camera=False, preview=True, CAN="", straight_cutoff = 0.05, corner_cutoff = 0.1, cornering_multiplier = 1):
    if camera:
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_sizes["camera"][0]) # test if 848 x 480 is available on streamcam
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_sizes["camera"][1])
        capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        capture.set(cv2.CAP_PROP_FOCUS, 0)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) #important to set right codec and enable the 60fps
        capture.set(cv2.CAP_PROP_FPS, 60) #enable 60FPS
    else:
        capture = cv2.VideoCapture(video)
        frame_sizes["camera"] = (capture.get(3), capture.get(4))

    # set up can connection and schedule periodic messages
    if CAN:
        command = 'sudo ip link set ' + CAN + ' type can bitrate 500000'
        os.system('echo %s|sudo -S %s' % (SUDO_PASSWORD, command))
        command = 'sudo ip link set ' + CAN + ' up'
        os.system('echo %s|sudo -S %s' % (SUDO_PASSWORD, command))

        bus = can.Bus(interface='socketcan', channel=CAN, bitrate=500000)

        homing_msg = can.Message(arbitration_id=0x6f1, data=[0, 0, 0, 0, 0, 0, 0, 0]) # probably not needed but saves time on startup as homing starts before model
        bus.send(homing_msg)

        steering_msg = can.Message(arbitration_id=0x12c, data=[0, 0, 0, 0, 0, 0, 0, 0])
        steering_task = bus.send_periodic(steering_msg, CAN_MSG_SENDING_SPEED)

        acc_msg = can.Message(arbitration_id=0x120, is_extended_id=True, data=[acc_speed, 0, 1, 0, 0, 0, 0, 0])
        acc_task = bus.send_periodic(acc_msg, CAN_MSG_SENDING_SPEED)

    try:
        while (capture.isOpened()):
            ok, frame_orig = capture.read()
            if not ok: break
            # prep image for torch model and get steering angle prediction
            resized_img = cv2.resize(frame_orig, (frame_sizes["model"][0], frame_sizes["model"][1]))
            normalized_img = (resized_img / 127 - 1).astype(np.float32)
            # normalized_img = normalized_img[160:325,0:848]
            transform = transforms.Compose([ToTensor()])
            transformed_img = transform(normalized_img)
            outputs = model(transformed_img.to(dev)).detach().cpu().numpy()
            steer = max(min(outputs[0][0], 0.9), -0.9)
            original_steer = steer

            # prep image for torch model and get steering direction prediction
            frame = cv2.resize(frame_orig, (frame_sizes["classification"][0], frame_sizes["classification"][1])).astype(np.float32)
            frame = (frame / 255).astype(np.float32)
            transform = transforms.Compose([ToTensor()])
            frame = transform(frame)
            predictions = classification_model(frame)

            p = F.softmax(predictions, dim=1)  # probabilities
            i = p.argmax()  # max index

            if i == 0: steer = (min(steer, corner_cutoff * -1)) * cornering_multiplier      # left
            elif i == 1: steer = (max(steer, corner_cutoff))                                # right
            else: steer = min(max(steer, straight_cutoff * -1), straight_cutoff)    

            classification_pred = "straight"
            if i == 0: 
                classification_pred = "  left"
            elif i == 1: classification_pred = "  right"

            print(f'Classification: {i} ({classification_pred}, {p[0, i]:.2f})\nPredicted steering:{original_steer} actual steering: {steer})')

            # send new steering value to CAN-bus if enabled
            if CAN:
                steering_msg.data = list(bytearray(struct.pack("f", float(steer))) + bytearray(4))
                steering_task.modify_data(steering_msg)

            # show live camera preview with prediction if enabled
            if preview:
                w, h = frame_sizes["camera"][0], frame_sizes["camera"][1]
                cv2.line(frame_orig, (int(w//2), int(h)), (int(w*(1+steer)//2), int(h//2)), (255, 0, 0), 2) # (blue) steering model with cap
                cv2.line(frame_orig, (int(w//2), int(h)), (int(w*(1+original_steer)//2), int(h//2)), (0, 0, 255), 2) # (red) steering model without cap
                cv2.putText(frame_orig, f'{classification_pred}', (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), thickness=4)
                cv2.imshow('Kartview', frame_orig)
                cv2.waitKey(1)

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    video = 'assets/videos/kwali.avi'
    dev = torch.device("cpu")

    model = SelfDriveModel(gpu=False)
    model.load_state_dict(torch.load(f"{model_name}", map_location=dev))
    model.eval()
    model.to(dev)

    main(model, classification_model, frame_sizes, acc_speed, camera, preview, CAN, straight_cutoff, corner_cutoff, cornering_multiplier)
    cv2.destroyAllWindows()
