import torch
from numpy import transpose
import numpy as np
from typing import Dict, Union

# Scuffed fix for using transforms on deployment in python. Leave on false unless you are using python_deploy_model.py
deploy = False

class ToTensor(object):
    """Convert ndarrays in sample to Tensors"""

    def __call__(self, sample):
        if deploy:
            sample = sample.transpose((2, 0, 1))
            sample = torch.from_numpy(sample)
            return sample
        else:
            image, actions = sample['image'], sample['actions']

            # Numpy images have a shape of HWC, tensor images have a shape of CHW
            image = image.transpose((2, 0, 1))
            return {'image': torch.from_numpy(image),
                    'actions': torch.from_numpy(actions)}


class Normalizer(object):
    """
    Normalize the array by subtracting mean and dividing by std
    """

    def __init__(self, mean, std):
        """
        :param mean: list of mean values (one for each channel)
        :param std: list of standard deviation values (one for each channel)
        """
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, sample: Dict[str, Union[np.ndarray, np.ndarray, float]]) -> Dict[str, Union[np.ndarray, np.ndarray, float]]:
        if deploy:
            return (sample.astype(np.float32) - self.mean) / self.std
        else:    
            image, actions = sample['image'], sample['actions']
            return {'image': ((image.astype(np.float32) - self.mean) / self.std), 'actions': actions}

        