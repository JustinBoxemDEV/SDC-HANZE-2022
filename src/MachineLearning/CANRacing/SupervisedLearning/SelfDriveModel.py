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
            nn.Linear(in_features=10, out_features=3)
        )

    def forward(self, x):
        x = x.view(1, 3, 480, 848) # x.size(0)
        output = self.conv_layers(x)
        output = output.view(1, -1) # output.size(0)
        output = self.linear_layers(output)

        output = output.type(torch.cuda.DoubleTensor) # comment this if you use CPU (for deployment) 
        return output

# --------------------------------------------------------
# For testing

# model = SelfDriveModel()
# print('The model:')
# print(model)

# print('\n\nModel params:')
# for param in model.parameters():
#     print(param.shape)
