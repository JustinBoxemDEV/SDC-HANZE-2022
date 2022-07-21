from TTAssenDataset import TTAssenDataset
import torchvision.transforms as transforms
import torch.utils.data
import albumentations as A
from transforms import ToTensor, Normalizer

def get_dataloader(img_folder: str, act_csv: str, batch_size: int, normalize=True, normalize_std=False, motion_blur=False, 
                    random_brightness_contrast=False, random_gamma=False, flip=False, rotate=False):
    """
    Returns a dataloader (pytorch) which can sample images from the dataset. This dataloader will perform data augmentations based on the passed parameters.

    :param img_folder Path to the folder containing the dataset images (Base folder for the "images xx-xx-xx" folders)
    :param act_csv Path to the csv containing the predictions with their corresponding image name in format: images 22-03-2022 13-26-11/1647952034.1168804.jpg
    :param batch_size The number of images in a batch
    :param normalize Boolean if the images should be normalized (only use 1 normalization at a time)
    :parm normalize_std Boolean if the images should be normalized using standard deviation (only use 1 normalization at a time)
    :param motion_blur Boolean if motion blur should be applied to the images in a sample based on a percentace chance
    :param random_brightness_contrast Boolean if a range of random brightness and contrast should be applied to the images in a sample based on a percentace chance
    :param random_gamma Boolean if a random gamma in a range should be applied to the images in a sample based on a percentace chance
    :param flip Boolean if images should randomly be flipped
    :param rotate Boolean if images should randomly be rotated
    """
    t = []
    albu_t = []

    # Add transforms here!
    if normalize_std:
        t.append(Normalizer(0, 255))
    if motion_blur:
        albu_t.append(A.MotionBlur(blur_limit=8, p=0.5))
    if random_brightness_contrast:
        albu_t.append(A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, brightness_by_max=True, p=0.5))
    if random_gamma:
        albu_t.append(A.RandomGamma(gamma_limit=(80, 120), p=0.5))

    # always convert to tensor
    t.append(ToTensor())

    dataset = TTAssenDataset(root_dir=img_folder, csv_file=act_csv, 
                                transforms=transforms.Compose(t), albu_transforms=A.Compose(albu_t), normalize=normalize, rotate=rotate, flip=flip)
                                
    dataloader = torch.utils.data.DataLoader(dataset, drop_last=True, num_workers=4, batch_size=batch_size, shuffle=True) # Change num_workers to a lower number if your device can't handle the load
    return dataloader