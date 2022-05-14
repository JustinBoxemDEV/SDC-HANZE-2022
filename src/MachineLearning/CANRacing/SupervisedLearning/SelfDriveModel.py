import torch.nn as nn
import torch

class SelfDriveModel(nn.Module):

    def __init__(self):
        super(SelfDriveModel, self).__init__()

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
            nn.Linear(in_features=64*234*418, out_features=64),
            nn.ELU(),
            # nn.Dropout(p=0.4),
            nn.Linear(in_features=64, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=3)
        )

    def forward(self, x):
        x = x.view(x.size(0), 3, 480, 848) # x.size(0) if training, 1 if deploying
        output = self.conv_layers(x)
        # print(output.shape)
        output = output.view(output.size(0), -1) # output.size(0) if training, 1 if deploying
        output = self.linear_layers(output)
        output = output.type(torch.cuda.DoubleTensor) # comment this if you use CPU (for deployment) TODO: dynamic fix
        return output

