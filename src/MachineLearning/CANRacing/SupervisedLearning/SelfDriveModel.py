import torch.nn as nn
import torch

class SelfDriveModel(nn.Module):
    def __init__(self, gpu=True):       # Temporary GPU parameter as lazy fix for training on gpu and testing on cpu
        super(SelfDriveModel, self).__init__()
        self.gpu = gpu 

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(36, 64, kernel_size=3, stride=1),
            # nn.ELU(),
            # nn.Dropout(p=0.5) # TODO: test if this helps
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64*77*418, out_features=64), # for full image use 64x234x418
            nn.ELU(),
            # nn.Dropout(p=0.4),
            nn.Linear(in_features=64, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=2)
        )

    def forward(self, x):
        if self.gpu:
            x = x.view(x.size(0), 3, 165, 848) # for full image use 3, 480, 848
        else:
            x = x.view(1, 3, 165, 848)

        output = self.conv_layers(x)
        # print(output.shape)

        if self.gpu:
            output = output.view(output.size(0), -1)
        else:
            output = output.view(1, -1)

        output = self.linear_layers(output)

        if self.gpu:
            output = output.type(torch.cuda.DoubleTensor)
        return output

