import torch
import pandas as pd
import os
import numpy as np
import cv2
from random import random
import albumentations as A

class TTAssenDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir: str, csv_file: str,
                    transforms=None, albu_transforms=None, normalize=True, flip=False, rotate=False):
        """
        :param root_dir: Path to folder containing directory with all the images
        :param csv_file: Path to the csv file with actions and corresponding image names
        :param transform (callable, optional) Optional transform to be applied on a sample
        :param albu_transforms (callable, optional) Optional trasform from the albumentations library to be applied on a sample
        :param flip
        :param rotate
        """
        self.actions_frames = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transforms = transforms
        self.albu_transforms = albu_transforms
        self.normalize = normalize
        self.flip = flip
        self.rotate = rotate
    
    def __len__(self):
        return len(self.actions_frames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.actions_frames.iloc[idx, -1])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (368, 207)) # resize images here!

        # extract ground truth steering from CSV
        steer = self.actions_frames.iloc[idx, 0]
        # throttle = self.actions_frames.iloc[idx, 1]
        # brake = self.actions_frames.iloc[idx, 2]
        
        if self.flip and random() > 0.5:
            image = cv2.flip(image, 1)
            steer = steer * -1

        if self.rotate and random() > 0.5:
            transform = A.Compose([A.Rotate(always_apply=False, p=0.5, limit=(-1, 1), interpolation=4, border_mode=4)])
            image = transform(image=image)['image']

        # Different method of normalizing, if you use this set normalize_std=False in load_data.py function
        if self.normalize:
            image = image / 127.5 -1
            image = image.astype(np.float32)

        actions = np.array([steer]) # throttle, brake

        # Slice the images to remove noise
        # sample = {'image': image[160:325,0:848], 'actions': actions}
        sample = {'image': image, 'actions': actions}

        # transforms which use the albumentations library
        if self.albu_transforms:
            sample['image'] = self.albu_transforms(image=sample['image'])['image']

        # torchvision transforms
        if self.transforms:
            sample = self.transforms(sample)

        sample['img_names'] = img_name
        
        return sample
