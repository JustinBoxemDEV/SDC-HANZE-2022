# Set deploy in transforms.py to True to use this!

from os import listdir
import cv2
from SelfDriveModel import SelfDriveModel
import torch
import numpy as np
from transforms import Normalizer, ToTensor
import torchvision.transforms as transforms

model_name = "SLSelfDriveModel_nobrake_2022-05-27_13-42-18"
dev = "cpu"

model = SelfDriveModel(gpu=False)
model.load_state_dict(torch.load(f"assets/models/{model_name}.pt", map_location=dev))
model.eval()
model.to(dev)

for image_name in listdir("D:/SDC/sdc_data/justin_data/original/images 30-03-2022 15-17-40/"):
    img = cv2.imread(f"D:/SDC/sdc_data/justin_data/original/images 30-03-2022 15-17-40/{image_name}")
    # print(img)

    # resize
    img = np.resize(img, (480, 848, 3)) # resize images here!
    
    # Slice the images to remove noise
    cropped_img = img[160:325,0:848]
    
    # transforms
    t = []
    t.append(Normalizer(0, 255))
    t.append(ToTensor())
    transform = transforms.Compose(t)
    normalized_cropped_img = transform(cropped_img)

    outputs = model(normalized_cropped_img.to(dev)).detach().cpu().numpy()
    steer, throttle = outputs[0][0], outputs[0][1]
    # print("steer:", steer)
    # print("throttle:", throttle)

    cv2.putText(img, f'{steer:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.rectangle(img, (50, 400), (70, int(400-(throttle))), (0, 0, 255), cv2.FILLED)
    cv2.line(img, (int(848//2), int(480)), (int(848*(1+steer)//2), int(480//2)), (0, 0, 255), 2)
    cv2.imshow('Prediction visualization', img)
    cv2.waitKey(1000//60)