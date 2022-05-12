import torch
import pandas as pd
import os
import skimage.io
import numpy as np


class TTAssenDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir: str, csv_file: str,
                    transforms=None, albu_transforms=None):
        """
        Param csv_file (string): Path to the csv file with actions
        Param root_dir (string): Path to folder containing directory with all the images
        Param transform (callable, optional) Optional transform to be applied on a sample
        """
        self.actions_frames = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transforms = transforms
        self.albu_transforms = albu_transforms
    
    def __len__(self):
        return len(self.actions_frames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.actions_frames.iloc[idx, 3])

        image = skimage.io.imread(img_name)

        steer = self.actions_frames.iloc[idx, 0]
        throttle = self.actions_frames.iloc[idx, 1]
        brake = self.actions_frames.iloc[idx, 2]

        actions = np.array([steer, throttle, brake])

        sample = {'image': image, 'actions': actions, 'img_names': img_name}

        # Augmentation transforms which use the albumentations library
        if self.albu_transforms:
            sample['image'] = self.albu_transforms(image=sample['image'])['image']

        if self.transforms:
            sample = self.transforms(sample)

        return sample
