import torch
import pandas as pd
import os
import skimage.io
import numpy as np
from matplotlib import pyplot as plt
from transforms import ToTensor

class TTAssenDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
    Param csv_file (string): Path to the csv file with annotations
    Param root_dir (string): Directory with all the images
    Param transform (callable, optional) Optional transform to be applied on a sample
    """
        self.actions_frames = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.actions_frames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.actions_frames.iloc[idx, 3])
        # print("img name:", img_name)

        image = skimage.io.imread(img_name)

        steer = self.actions_frames.iloc[idx, 0]
        throttle = self.actions_frames.iloc[idx, 1]
        brake = self.actions_frames.iloc[idx, 2]

        actions = np.array([steer, throttle, brake])

        sample = {'image': image, 'actions': actions}

        if self.transform:
            sample = self.transform(sample)

        return sample

# ----------------------------------------------------------------------------------------------------
# For testing

ttAssen_dataset = TTAssenDataset(csv_file='C:/Users/Sabin/Documents/SDC/RDW_Data/douwe_data_images_18-11-2021_14-59-21_2.csv',
                                    root_dir='C:/Users/Sabin/Documents/SDC/RDW_Data/')

fig = plt.figure()

for i in range(len(ttAssen_dataset)):
    sample = ttAssen_dataset[i]

    # print(sample)
    # print(i, sample['image'].shape, sample['actions'].shape)

    ax = plt.subplot(1, 4, i + 1) # show first 4 samples
    plt.tight_layout()
    ax.set_title(f'Sample #{i}')
    # ax.axis('off')
    ax.set_xlabel(f"Actions:{sample['actions']}")
    plt.imshow(sample['image'])

    if i == 3:
        plt.show()
        break
