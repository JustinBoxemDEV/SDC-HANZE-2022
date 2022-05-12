import torch
import skimage.io
from matplotlib import pyplot as plt
from load_data import get_dataloader
import tqdm
import numpy as np

@torch.no_grad()
def run_training(train_img_dir: str, train_actions_csv: str, valid_img_dir: str, valid_actions_csv: str, 
                num_epochs: int = 5, batch_size: int = 8, dev: str = "cuda:0"):
                
    train_loader = get_dataloader(img_folder=train_img_dir, act_csv=train_actions_csv, batch_size=batch_size, random_sun_flare=True)
    # valid_loader = get_dataloader(img_folder=valid_img_dir, act_csv=valid_actions_csv, batch_size=batch_size)

    for i, batch in enumerate(train_loader): # dataloader contains all images
        plt.figure(f"Image index:{i}")

        image = np.array(batch['image'][0])

        skimage.io.imshow(image)
        plt.show()

        # print("Image:", batch['image'])
        # print("Actions:", batch['actions'])
        # print(f"Index: {i}")
        # print("Image names:", batch['img_names']) # contains 8 images if batch_size = 8

    model = None

    optimizer = adam
    scheduler = optimizer

    print("Training...")
    for epoch in range(0, num_epochs):
        print(f"Executing epoch {epoch}!")

        for idx, batch in enumerate(train_loader):
            print(f"Executing batch number {idx}!")

            print("Images in batch:", batch['image'])
            print("Ground truth actions in batch:", batch['actions'])

            # input_data, actions = batch['image'].to(dev), batch['actions'].to(dev)

            # steer, throttle, brake = (0.0, 0, 0) # TODO: model predictions

            # loss = 0 # TODO: calculate loss

            # back prop?

            # loss_sum += loss
            # loss_cnt+=1
        
        # TODO: calc avg loss
        # TODO: tune learning rate?
        if epoch % 2 == 0:
            # run_validation(valid_loader=valid_loader)
            pass
    return


@torch.no_grad()
def run_testing(test_img_dir: str, test_actions_csv: str):
    test_loader = get_dataloader(img_folder=test_img_dir, act_csv=test_actions_csv, batch_size=8)

    model = None

    print("Testing...")
    for idx, batch in enumerate(test_loader):
        print(f"Executing batch number {idx}!")

        print("Images in batch:", batch['image'])
        print("Ground truth actions in batch:", batch['actions'])
        
        # steer, throttle, brake = (0.0, 0, 0) # TODO: model prediction
    return


@torch.no_grad()
def run_validation(valid_loader):
    print("there is no validation")
    pass


def run(training=False):
    if training:
        run_training(train_img_dir="C:/Users/Sabin/Documents/SDC/SL_data/training/", 
                    train_actions_csv="C:/Users/Sabin/Documents/SDC/SL_data/training/douwe_data_images_18-11-2021_14-59-21_2.csv", 
                    valid_img_dir="", valid_actions_csv="", # dont have validation images right now (maybe data images 18-11-2021 11-07-14)
                    num_epochs=1, batch_size=32, dev="cuda:0")

    # try to free up GPU memory
    torch.cuda.empty_cache()
    run_testing(test_img_dir="C:/Users/Sabin/Documents/SDC/SL_data/testing/", 
                test_actions_csv="C:/Users/Sabin/Documents/SDC/SL_data/testing/douwe_data_images_18-11-2021 15-12-21.csv")

    print("Done!")


if __name__ == "__main__":
    run_training(train_img_dir="D:\\SL_data\\training", 
                     train_actions_csv="D:\\SL_data\\training\\data images 18-11-2021 14-59-21.csv", 
                     valid_img_dir="", valid_actions_csv="", # dont have validation images right now (maybe data images 18-11-2021 11-07-14)
                     num_epochs=1, batch_size=32, dev="cuda:0")