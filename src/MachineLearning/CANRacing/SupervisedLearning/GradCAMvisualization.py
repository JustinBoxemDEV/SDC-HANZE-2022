# stolen from: https://medium.com/the-owl/gradcam-in-pytorch-7b700caa79e5

from traceback import print_tb
from SelfDriveModel import SelfDriveModel
from torch import nn
import cv2
from transforms import Normalizer, ToTensor
from torchvision import transforms
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from torchvision import models
from skimage.io import imread
from skimage.transform import resize

class GradCamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        
        #PRETRAINED MODEL
        self.pretrained = SelfDriveModel(gpu=False)
        self.layerhook.append(self.pretrained.linear_layers.register_forward_hook(self.forward_hook()))
        
        for p in self.pretrained.parameters():
            p.requires_grad = True
    
    def activations_hook(self,grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
        return hook

    def forward(self,x):
        # print(x.shape)
        out = self.pretrained(x)
        return out, self.selected_out


image = cv2.imread("C:/Users/Sabin/Documents/vsc_cpp_projects/SDC-stuff/SDC-HANZE-2022/assets/images/load-examples/1649757843.2887032.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (848, 480)) # resize images here!
image = image[160:325,0:848]

# transforms
t = []
t.append(Normalizer(0, 255))
t.append(ToTensor())
transform = transforms.Compose(t)
frame = transform(image)

# print(frame.shape)

gcmodel = GradCamModel()

outputs, acts = gcmodel(frame)

# loss_fn = torch.nn.MSELoss()
# # (out, torch.from_numpy(np.array([600])).to(‘cuda:0’))
# loss.backward()

grads = gcmodel.get_act_grads() # TODO: grads is none :(
print(grads)
pooled_grads = torch.mean(grads, dim=[0,2,3])

for i in range(acts.shape[1]):
    acts[:,i,:,:] += pooled_grads[i]

heatmap_j = torch.mean(acts, dim = 1).squeeze()
heatmap_j_max = heatmap_j.max(axis = 0)[0]
heatmap_j /= heatmap_j_max

heatmap_j = cv2.resize(heatmap_j, (224,224), preserve_range=True)

cmap = mpl.cm.get_cmap('jet', 256)
heatmap_j2 = cmap(heatmap_j,alpha = 0.2)

fig, axs = plt.subplots(1,1,figsize = (5,5))
axs.imshow((frame)[0].transpose(1,2,0))
axs.imshow(heatmap_j2)
plt.show()

for h in gcmodel.layerhook:
    h.remove()
for h in gcmodel.tensorhook:
    h.remove()