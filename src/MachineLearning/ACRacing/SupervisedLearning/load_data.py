from TTAssenDataset import TTAssenDataset
import torchvision.transforms as transforms
import torch.utils.data
import albumentations as A
from transforms import ToTensor

def get_dataloader(img_folder: str, act_csv: str, batch_size: int):
    t = []
    albu_t = []

    # Add transforms here!
    # t.append(ToTensor())
    # albu_t.append(A.RandomSunFlare(flare_roi=(0, 0, 1, 0.25), angle_lower=0, angle_upper=1, 
    #                                     num_flare_circles_lower=1, num_flare_circles_upper=5, src_radius=200, p=1))

    dataset = TTAssenDataset(root_dir=img_folder, csv_file=act_csv, 
                                transforms=transforms.Compose(t), albu_transforms=A.Compose(albu_t))
                                
    dataloader = torch.utils.data.DataLoader(dataset, drop_last=True, num_workers=0, batch_size=batch_size)
    return dataloader
