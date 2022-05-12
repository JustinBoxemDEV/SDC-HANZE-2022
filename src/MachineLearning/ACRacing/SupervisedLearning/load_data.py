from TTAssenDataset import TTAssenDataset
import torchvision.transforms as transforms
import torch.utils.data
import albumentations as A
from transforms import ToTensor, Normalizer

def get_dataloader(img_folder: str, act_csv: str, batch_size: int, normalize=True,
                    random_sun_flare=False, horizontal_flip=False, motion_blur=False, random_shadow=False, 
                    random_brightness_contrast=False, random_gamma=False):
    t = []
    albu_t = []

    # Add transforms here!
    if normalize:
        t.append(Normalizer(0, 255))
    if(random_sun_flare):
        albu_t.append(A.RandomSunFlare(flare_roi=(0, 0, 1, 0.25), angle_lower=0, angle_upper=1,num_flare_circles_lower=1, num_flare_circles_upper=5, src_radius=350, p=1))
    if(horizontal_flip):
        albu_t.append(A.HorizontalFlip(p=0.5))
    if(motion_blur):
        albu_t.append(A.MotionBlur(blur_limit=10, p=0.5))
    if(random_shadow):
        albu_t.append(A.RandomShadow(shadow_roi=(0, 0.55, 1, 1), num_shadows_lower=0, num_shadows_upper=3, p=0.5, shadow_dimension=85))
    if(random_brightness_contrast):
        albu_t.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=0.5))
    if(random_gamma):
        albu_t.append(A.RandomGamma(gamma_limit=(80, 120), p=0.5))

    t.append(ToTensor())
    dataset = TTAssenDataset(root_dir=img_folder, csv_file=act_csv, 
                                transforms=transforms.Compose(t), albu_transforms=A.Compose(albu_t))
                                
    dataloader = torch.utils.data.DataLoader(dataset, drop_last=True, num_workers=0, batch_size=batch_size)
    return dataloader
