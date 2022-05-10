import torch.nn as nn
import cv2
import torch

class TinyTestModel(nn.Module):

    def __init__(self):
        super(TinyTestModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Dropout(p=0.5)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64*1*16, out_features=100),
            nn.ELU(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=100, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, x):
        print("X shape:", x.shape)
        
        x = x.view(x.size(0), 3, 66, 200)
        output = self.conv_layers(x)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output

model = TinyTestModel()

img = cv2.imread("src/MachineLearning/ACRacing/TestImges/ac480p.png")
# img = img[30:510, 10:650]

# cv2.imshow("img", img)
# cv2.waitKey(0)
img = torch.tensor(img)
print("Img shape:", img.shape)
print("Img size:", img.size())
output = model.forward(img)

print(output)

# TODO: Fix shape (shape '[509, 3, 66, 200]' is invalid for input of size 972699)

# ------------------------------------------------------
# print('The model:')
# print(model)

# print('\n\nModel params:')
# for param in model.parameters():
#     print(param)
