import torch
from numpy import transpose
import numpy as np
from typing import Dict, Union

# SCUFFED FIX FOR USING TRANSFORMS ON DEPLOYMENT IN PYTHON
deploy = True

class ToTensor(object):
    """Convert ndarrays in sample to Tensors for GPU acceleration"""

    def __call__(self, sample):
        if deploy:
            sample = sample.transpose((2, 0, 1))
            sample = torch.from_numpy(sample)
            return sample
        else:
            image, actions = sample['image'], sample['actions']

            # swap color axis because
            # numpy image: H x W x C
            # Tensor image: C x H x W
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
        :param std: list of stddev values (one for each channel)
        """
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, sample: Dict[str, Union[np.ndarray, np.ndarray, float]]) -> Dict[str, Union[np.ndarray, np.ndarray, float]]:
        """
        Normalize the input array

        :param sample: a dictionary with the 'images' array of shape [b, h, w, c]
        and the 'actions' array of shape [b, N, 5] for N actions

        :return: A dictionary with converted 'image', 'actions'
        """
        if deploy:
            return (sample.astype(np.float32) - self.mean) / self.std
        else:    
            image, actions = sample['image'], sample['actions']
            return {'image': ((image.astype(np.float32) - self.mean) / self.std), 'actions': actions}

        