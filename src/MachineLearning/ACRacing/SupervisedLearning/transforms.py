import torch

class ToTensor(object):
    """Convert ndarrays in sample to Tensors for GPU acceleration"""

    def __call__(self, sample):
        image, actions = sample['image'], sample['actions']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'actions': torch.from_numpy(actions)}
