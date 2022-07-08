""" To use the Python model in C++ it needs to be traced. 
    This script will trace the DirectionClassificationModel and save a copy as traced_{modelname}.pt 
"""
import torch
import numpy as np
import cv2
from DirectionClassificationModel import DirectionClassificationModel

image_path = "./assets/images/tracing_example_image.jpg"
model_name = "best"
normalize = lambda x, mean=0.5, std=0.25: (x - mean) / std

device = torch.device("cpu")
model = DirectionClassificationModel(gpu=False)
model.load_state_dict(torch.load(f"./assets/models/classification/{model_name}.pt", 
                      map_location=device))
model.to(device)

original_im = cv2.imread(image_path)[..., ::-1]

frame = cv2.resize(original_im, (128, 128)).astype(np.float32)
frame = (frame / 255).astype(np.float32)
frame = frame.transpose((2, 0, 1))
frame = torch.from_numpy(frame)

frame = frame.to(device=device)

model_traced = torch.jit.trace(model, frame)
model_traced.save(f'./assets/models/traced_{model_name}')

print("Done")