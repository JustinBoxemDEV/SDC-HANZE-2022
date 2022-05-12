import torch.nn as nn
import skimage.io
import matplotlib.pyplot as plt
import torch
import numpy as np
from transforms import Normalizer
from torchvision import transforms

class SelfDriveModel(nn.Module):

    def __init__(self):
        super(SelfDriveModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=3, stride=1)
            # nn.ELU(),
            # nn.Dropout(p=0.5)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=48*234*418, out_features=100),
            nn.ELU(),
            # nn.Dropout(p=0.4),
            nn.Linear(in_features=100, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, x):
        print("X shape:", x.shape)
        
        x = x.view(x.size(0), 3, 480, 848)
        print("X shape after transform:", x.shape)
        output = self.conv_layers(x)
        print(output.shape)
        output = output.view(output.size(0), -1)
        print("X shape after second transform:", x.shape)
        output = self.linear_layers(output)
        return output

# --------------------------------------------------------
# For testing
# Input image shape: (480, 848, 3)


# fig = plt.figure("Dataset samples")

# model = SelfDriveModel()
# img = skimage.io.imread("C:/Users/Sabin/Documents/SDC/SL_data/training/images 18-11-2021 14-59-21/1637243973.1105928.jpg")
# # skimage.io.imshow(img)


# # plt.show()
# img = np.array(img)
# print("Img shape:", img.shape)

# # ToTensor
# img = img.transpose((2, 0, 1))
# img = torch.from_numpy(img)
# print("Img shape after transform:", img.shape)

# print(img)

# # TODO: normalize data
# # t = Normalizer((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# # transforms.Compose(t)
# # t(img)

# output = model(img)

# print(output)

# TODO: Fix shape (shape '[509, 3, 66, 200]' is invalid for input of size 972699)

# ------------------------------------------------------
# model = SelfDriveModel()
# print('The model:')
# print(model)

# print('\n\nModel params:')
# for param in model.parameters():
#     print(param.shape)
