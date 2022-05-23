import torch
from SelfDriveModel import SelfDriveModel
from PIL import Image, ImageGrab
import numpy as np
from transforms import Normalizer, ToTensor
import torchvision.transforms as transforms
import skimage.io
import cv2

MODEL = "SLSelfDriveModel_2022-05-23_00-55-20_Adam_0.00001.pt"
IMAGE = "tracing_example_image.jpg"

device = torch.device("cpu")
model = SelfDriveModel(gpu=False)

model.load_state_dict(torch.load(f"./assets/models/{MODEL}", 
                      map_location=device))
model.to(device)

np_image = skimage.io.imread(f'./assets/images/{IMAGE}') 

np_image = np.resize(np_image, (480, 848, 3)).astype(np.float32)
np_image = np_image[160:325,0:848] # yy xx cv::Rect(xMin,yMin,xMax-xMin,yMax-yMin) 

cv2.imshow('image',np_image)
print(np_image.shape)

transforms
t = []
t.append(Normalizer(0, 255))
t.append(ToTensor())
transform = transforms.Compose(t)
example_image = transform(np_image)

example_image = example_image.to(device=device)

model_traced = torch.jit.trace(model, example_image)
model_traced.save(f'./assets/models/traced_{MODEL}')
