# Set deploy in transforms.py to True to use this!
# If you are using an old model (that includes brake in the model) you may have to change out_features in SelfDriveModel.py at line 48 to 3.

from os import listdir
import cv2
import torch
import numpy as np
import time
import torch.nn.functional as F

# Functions
normalize = lambda x, mean=0.5, std=0.25: (x - mean) / std
denormalize = lambda x, mean=0.5, std=0.25: x * std + mean

model_name = "exp"
image_folder_path = "C:/Users/Sabin/Documents/SDC/SL_data/Visualization_test/images 30-03-2022 15-17-40/"

dev = "cpu"
model = torch.load(f'src/MachineLearning/CANRacing/SupervisedLearning/runs/{model_name}/weights/best.pt', 
        map_location=torch.device('cpu'))['model'].float()

# model.load_state_dict(torch.load(f"src/MachineLearning/CANRacing/SupervisedLearning/runs/{model_name}/weights/best.pt", map_location=dev))
model.eval()
model.to(dev)

for image in listdir(image_folder_path):
    start = time.time()

    # TODO: extract ground truth

    image_path = image_folder_path + image

    original_im = cv2.imread(str(image_path))[..., ::-1]
    # print(original_im)

    # resize
    resize = torch.nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)
    
    # transforms
    im = np.ascontiguousarray(np.asarray(original_im).transpose((2, 0, 1)))  # HWC to CHW
    im = torch.tensor(im).float().unsqueeze(0) / 255.0  # to Tensor, to BCWH, rescale
    resized_image = resize(normalize(im))

    # make prediction
    predictions = model(im)

    p = F.softmax(predictions, dim=1)  # probabilities
    i = p.argmax()  # max index
    # print(f'{image} prediction: {i} ({p[0, i]:.2f})')

    steer = "straight"
    if i == 0: steer = "  left"
    elif i == 1: steer = "  right"
    # print(steer)

    original_im = cv2.cvtColor(original_im, cv2.COLOR_BGR2RGB)

    # visualize (with original image)
    cv2.putText(original_im, f'{steer}', (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 100, 255), thickness=4)
    cv2.imshow('Kartview', original_im)
    # cv2.waitKey(max(0, int((time.time()-start)*1000)))
    cv2.waitKey(1000//60)

print("Done!")
