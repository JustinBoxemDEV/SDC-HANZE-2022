from load_data import get_dataloader
import skimage.io
from matplotlib import pyplot as plt

dataloader = get_dataloader(img_folder="C:/Users/Sabin/Documents/SDC/SL_data/training/", 
                            act_csv="C:/Users/Sabin/Documents/SDC/SL_data/training/douwe_data_images_18-11-2021_14-59-21_2.csv", 
                            batch_size=8)

for i, batch in enumerate(dataloader): # dataloader contains all images
    # plt.figure(f"Image index:{i}")
    
    # img_name = batch['img_names']
    # img = skimage.io.imread(img_name[0]) # first image in the batch

    # skimage.io.imshow(img)
    # plt.show()

    # print("Image:", batch['image'])
    # print("Actions:", batch['actions'])
    print(f"Index: {i}")
    print("Image names:", batch['img_names']) # contains 8 images if batch_size = 8
   