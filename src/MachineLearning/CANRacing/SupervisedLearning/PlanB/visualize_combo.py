# Set deploy in transforms.py to True to use this!
# If you are using an old model (that includes brake in the model) you may have to change some things in SelfDriveModel.py.

import cv2
from SelfDriveModel import SelfDriveModel
from DirectionClassificationModel import DirectionClassificationModel
import torch
import numpy as np
from transforms import ToTensor
import torchvision.transforms as transforms
import pandas as pd
import torch.nn.functional as F

straight_cutoff = 0.1 # for swerving
corner_cutoff = 0.28 # for steering too much/little
cornering_multiplier = 1.05

model_name = "seed4_368-207_SteerSLSelfDriveModel_2022-06-07_00-03-49"
csv_file_path = "D:/KWALI/sorted_data images 07-06-2022 16-37-30.csv"

classification_model = DirectionClassificationModel(gpu=False)
classification_model.load_state_dict(torch.load(f"assets/models/classification/best.pt", map_location=torch.device('cpu')))

dev = "cpu"
model = SelfDriveModel(gpu=False)
model.load_state_dict(torch.load(f"./assets/models/{model_name}.pt", map_location=dev))
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

    frame_orig = cv2.imread(f"D:/KWALI/{i_name}")

    # Steering prediction model
    resized_img = cv2.resize(frame_orig, (368, 207))
    normalized_img = (resized_img / 127 - 1).astype(np.float32)
    # normalized_img = normalized_img[160:325,0:848]
    
    # Transform
    transform = transforms.Compose([ToTensor()])
    transformed_img = transform(normalized_img)

    # Predict steering
    outputs = model(transformed_img.to(dev)).detach().cpu().numpy()
    steer = max(min(outputs[0][0], 0.9), -0.9)
    original_steer = steer

    # Classificaiton model
    resized_img = cv2.resize(frame_orig, (128, 128)).astype(np.float32)
    normalized_img = (resized_img / 255).astype(np.float32)

    # Transform
    transform = transforms.Compose([ToTensor()])
    transformed_img = transform(normalized_img)

    # Predict classificatie
    predictions = classification_model(transformed_img)
    p = F.softmax(predictions, dim=1)  # probabilities
    i = p.argmax()

    if i == 0: steer = (min(original_steer, corner_cutoff * -1)) * cornering_multiplier      # left
    elif i == 1: steer = (max(original_steer, corner_cutoff))                                # right
    else: steer = min(max(original_steer, straight_cutoff * -1), straight_cutoff)            # straight

    classification_pred = "straight"
    if i == 0: 
        classification_pred = "  left"
    elif i == 1: classification_pred = "  right"

    # print(f'Classification: {i} ({classification_pred}, {p[0, i]:.2f})\nPredicted steering:{original_steer} actual steering: {steer})')

    # visualize (with original image)
    cv2.putText(frame_orig, f'{steer:.2f}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    cv2.putText(frame_orig, f'{truth_steer:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    cv2.line(frame_orig, (int(848//2), int(480)), (int(848*(1+original_steer)//2), int(480//2)), (0, 0, 255), 2) # (red) steering model without cap
    cv2.line(frame_orig, (int(848//2), int(480)), (int(848*(1+steer)//2), int(480//2)), (255, 0, 0), 2) # (blue) steering model with cap
    cv2.line(frame_orig, (int(848//2), int(480)), (int(848*(1+truth_steer)//2), int(480//2)), (0, 255, 0), 2)  # (green) ground truth steer
    cv2.putText(frame_orig, f'{classification_pred}', (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), thickness=4) # (blue) classification model prediction
    cv2.imshow('Prediction visualization', frame_orig)
    cv2.waitKey(1000//60)

print("Done!")