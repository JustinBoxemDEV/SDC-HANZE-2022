# Set deploy in transforms.py to True to use this!
# If you are using an old model (that includes brake in the model) you may have to change out_features in SelfDriveModel.py at line 48 to 3.

import cv2
from SelfDriveModel import SelfDriveModel
import torch
import numpy as np
from transforms import Normalizer, ToTensor
import torchvision.transforms as transforms
import pandas as pd


model_name = "final_seed400_SteerSLSelfDriveModel_2022-06-05_19-51-22"
csv_file_path = "C:/Users/Sabin/Documents/SDC/SL_data/testing/testing_60p_100_new_data_images_30-03-2022_15-17-40_smoothed.csv"

dev = "cpu"
model = SelfDriveModel(gpu=False)
model.load_state_dict(torch.load(f"/assets/models/{model_name}.pt", map_location=dev))
model.eval()
model.to(dev)

# count the amount of lines in the csv
with open(csv_file_path) as f:
    row_count = sum(1 for line in f)

actions_frames = pd.read_csv(csv_file_path)
for i in range(row_count-1):
    # extract ground truth
    truth_steer = actions_frames.iloc[i, 0]
    truth_throttle = actions_frames.iloc[i, 1]
    i_name = actions_frames.iloc[i, 3]
    
    img = cv2.imread(f"C:/Users/Sabin/Documents/SDC/SL_data/testing/{i_name}")

    # resize and crop
    img = np.resize(img, (480, 848, 3)) 
    cropped_img = img[160:325,0:848]
    
    # transforms
    t = []
    t.append(Normalizer(0, 255))
    t.append(ToTensor())
    transform = transforms.Compose(t)
    normalized_cropped_img = transform(cropped_img)

    # make prediction
    outputs = model(normalized_cropped_img.to(dev)).detach().cpu().numpy()
    # steer, throttle = min(outputs[0][0], 1.0), min(outputs[0][1], 100)
    steer = min(outputs[0][0], 1.0)
    steer = max(steer, -1)
    # throttle = max(throttle, 0)

    # visualize (with original image)
    cv2.putText(img, f'{steer:.2f}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.putText(img, f'{truth_steer:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    # cv2.rectangle(img, (45, 400), (65, int(400-(throttle))), (255, 0, 0), cv2.FILLED)
    # cv2.rectangle(img, (40, 400), (20, int(400-(truth_throttle))), (0, 255, 0), cv2.FILLED)
    cv2.line(img, (int(848//2), int(480)), (int(848*(1+steer)//2), int(480//2)), (0, 0, 255), 2)
    cv2.line(img, (int(848//2), int(480)), (int(848*(1+truth_steer)//2), int(480//2)), (0, 255, 0), 2)
    cv2.imshow('Prediction visualization', img)
    cv2.waitKey(1000//60)

print("Done!")