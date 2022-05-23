import torch
from SelfDriveModel import SelfDriveModel
from PIL import Image, ImageGrab
import numpy as np
from transforms import Normalizer, ToTensor
import torchvision.transforms as transforms
import skimage.io

MODEL = "SLSelfDriveModel90p.pt"
IMAGE = "tracing_example_image.jpg"

device = torch.device("cpu")
model = SelfDriveModel()

model.load_state_dict(torch.load(f"./assets/models/{MODEL}", 
                      map_location=device))
model.to(device)

np_image = skimage.io.imread(f'./assets/images/{IMAGE}') 

np_image = np.resize(np_image, (480, 848, 3)).astype(np.float32)

transforms
t = []
t.append(Normalizer(0, 255))
t.append(ToTensor())
transform = transforms.Compose(t)
example_image = transform(np_image)

example_image = example_image.to(device=device)

model_traced = torch.jit.trace(model, example_image)
model_traced.save(f'./assets/models/traced_{MODEL}')
