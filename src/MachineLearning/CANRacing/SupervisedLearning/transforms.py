import torch
from numpy import transpose
import numpy as np
from typing import Dict, Union

class ToTensor(object):
    """Convert ndarrays in sample to Tensors for GPU acceleration"""

    def __call__(self, sample):
        image, actions = sample['image'], sample['actions']

        # swap color axis because
        # numpy image: H x W x C
        # Tensor image: C x H x W
        # print("Image in totensor:", image)
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
        # if mean is None:
        #    mean = [[[0.485, 0.456, 0.406]]]
        # if std is None:
        #    std = [[[0.229, 0.224, 0.225]]]
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, sample: Dict[str, Union[np.ndarray, np.ndarray, float]]) -> Dict[str, Union[np.ndarray, np.ndarray, float]]:
        """
        Normalize the input array

        :param sample: a dictionary with the 'images' array of shape [b, h, w, c]
        and the 'actions' array of shape [b, N, 5] containing [y1, x1, y2, x2, class_id] for N actions

        :return: A dictionary with converted 'image', 'actions'
        """
        image, actions = sample['image'], sample['actions']
        return {'image': ((image.astype(np.float32) - self.mean) / self.std), 'actions': actions}