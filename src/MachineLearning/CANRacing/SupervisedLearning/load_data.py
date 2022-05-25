from TTAssenDataset import TTAssenDataset
import torchvision.transforms as transforms
import torch.utils.data
import albumentations as A
from transforms import ToTensor, Normalizer

def get_dataloader(img_folder: str, act_csv: str, batch_size: int, normalize=False,
                    random_sun_flare=False, horizontal_flip=False, motion_blur=False, random_shadow=False, 
                    random_brightness_contrast=False, random_gamma=False):
    """
    Returns a dataloader (pytorch) which can sample images from the dataset. This dataloader will perform data augmentations based on the passed parameters.

    :param img_folder The folder containing the database images
    :param act_csv The csv containing the actions with their corresponding image name
    :param batch_size The size of each batch that will be sampled at a time
    :parm normalize Boolean if the images should be normalized (reccomended to keep this on True)
    :parm random_sun_flare Boolean if a random sun flare should be added to the images in a sample based on a percentace chance
    :param horizontal_flip Boolean if the images in a sample should be horizontally flipped based on a percentace chance (actions need to be inverted if this is used, not implemented yet)
    :param motion_blur Boolean if motion blur should be applied to the images in a sample based on a percentace chance
    :param random_shadow Boolean if a random shadow should be applied to the images in a sample based on a percentace chance
    :param random_brightness_contrast Boolean if a range of random brightness and contrast should be applied to the images in a sample based on a percentace chance
    :param random_gamma Boolean if a random gamma in a range should be applied to the images in a sample based on a percentace chance
    """
    t = []
    albu_t = []

    # Add transforms here!
    if normalize:
        t.append(Normalizer(0, 255))
    if(motion_blur):
        albu_t.append(A.MotionBlur(blur_limit=8, p=0.5))
    # if(random_shadow):
    #     albu_t.append(A.RandomShadow(shadow_roi=(0, 0.55, 1, 1), num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=5, p=0.5))
    if(random_brightness_contrast):
        albu_t.append(A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, brightness_by_max=True, p=0.5))
    if(random_gamma):
        albu_t.append(A.RandomGamma(gamma_limit=(80, 120), p=0.5))

    # To use GPU acceleration, always convert to tensor
    t.append(ToTensor())
    dataset = TTAssenDataset(root_dir=img_folder, csv_file=act_csv, 
                                transforms=transforms.Compose(t), albu_transforms=A.Compose(albu_t))
                                
    dataloader = torch.utils.data.DataLoader(dataset, drop_last=True, num_workers=8, batch_size=batch_size, shuffle=True) # num_workers veranderen naar een lager getal als je device het niet aan kan
    return dataloader
