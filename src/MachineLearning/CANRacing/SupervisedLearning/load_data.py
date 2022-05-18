from TTAssenDataset import TTAssenDataset
import torchvision.transforms as transforms
import torch.utils.data
import albumentations as A
from transforms import ToTensor, Normalizer

def get_dataloader(img_folder: str, act_csv: str, batch_size: int, normalize=False,
                    random_sun_flare=False, horizontal_flip=False, motion_blur=False, random_shadow=False, 
                    random_brightness_contrast=False, random_gamma=False):
    """
    Returns a dataloader (pytorch) which can sample images from the dataset

    :param img_folder The folder containing the database images
    :param act_csv The csv containing the actions with their corresponding image name
    :param batch_size The size of each batch that will be sampled at a time
    :parm normalize Normalize the images (reccomended to keep this on True)
    :parm random_sun_flare Add a random sun flare to the images in the sample based on a percentace chance
    :param horizontal_flip Horizontally flip the images in the sample based on a percentace chance (actions need to be inverted if this is used, not implemented yet)
    :param motion_blur Apply motion blur to the images in the sample based on a percentace chance
    :param random_shadow Apply a random shadow to the images in the sample based on a percentace chance
    :param random_brightness_contrast Apply a random brightness and contrast in a range to the images in the sample based on a percentace chance
    :param random_gamma Apply a random gamma in a range to the images in the sample based on a percentace chance
    """
    t = []
    albu_t = []

    # Add transforms here!
    if normalize:
        t.append(Normalizer(0, 255))
    if(random_sun_flare):
        albu_t.append(A.RandomSunFlare(flare_roi=(0, 0, 1, 0.25), angle_lower=0, angle_upper=1,num_flare_circles_lower=1, num_flare_circles_upper=5, src_radius=350, p=0.5))
    if(horizontal_flip):
        albu_t.append(A.HorizontalFlip(p=0.5))
    if(motion_blur):
        albu_t.append(A.MotionBlur(blur_limit=10, p=0.5))
    if(random_shadow):
        albu_t.append(A.RandomShadow(shadow_roi=(0, 0.55, 1, 1), num_shadows_lower=0, num_shadows_upper=3, shadow_dimension=85, p=0.5))
    if(random_brightness_contrast):
        albu_t.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=0.5))
    if(random_gamma):
        albu_t.append(A.RandomGamma(gamma_limit=(80, 120), p=0.5))

    # To use GPU acceleration, always convert to tensor
    t.append(ToTensor())
    dataset = TTAssenDataset(root_dir=img_folder, csv_file=act_csv, 
                                transforms=transforms.Compose(t), albu_transforms=A.Compose(albu_t))
                                
    dataloader = torch.utils.data.DataLoader(dataset, drop_last=True, num_workers=0, batch_size=batch_size, shuffle=True)
    return dataloader
