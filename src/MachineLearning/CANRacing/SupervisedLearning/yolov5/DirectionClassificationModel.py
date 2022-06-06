import torch.nn as nn

class DirectionClassificationModel(nn.Module):
    def __init__(self, gpu=False):       # Temporary GPU parameter as lazy fix for training on gpu and testing on cpu
        super(DirectionClassificationModel, self).__init__()
        self.gpu = gpu 

        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1),
            nn.Flatten(),
            nn.Linear(in_features=1722368, out_features=3),
        )

    def forward(self, x):
        if self.gpu:
            x = x.view(x.size(0), 3, 128, 128)
        else:
            x = x.view(1, 3, 128, 128)
        output = self.layers(x)
        return output
