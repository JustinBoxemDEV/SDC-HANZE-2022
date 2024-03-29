"""
This pipeline is an implementation of a self driving pipeline using pytorch. Inspired by the original pytorch documentation/tutorials.
Training is done on the GPU and testing is done on the CPU (adjustable).

SETUP THE PIPELINE:
1. Create a conda environment with python 3.7.13 
    and run: conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch  
    (to prevent "Torch not compiled with CUDA enabled", this step is only needed if you plan on training on GPU)
2. Then run pip install requirements.txt

Sources:
https://pytorch.org/docs/stable/index.html
https://pytorch.org/docs/stable/tensorboard.html
https://pytorch.org/docs/stable/amp.html
https://github.com/bndr/pipreqs
"""

# TODO: 
# Somehow limit NN output values https://discuss.pytorch.org/t/how-to-return-output-values-only-from-0-to-1/24517/5

import torch
from load_data import get_dataloader
from tqdm import tqdm
from SelfDriveModel import SelfDriveModel
from utilities import static_var, wait_forever
import numpy as np
from tensorboard_visualize import create_tb, tb_show_text, tb_show_loss, tb_show_image, draw_pred_and_target_npy
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import cv2
torch.manual_seed(4)

def run_training(train_img_dir: str, train_actions_csv: str, valid_img_dir: str, valid_actions_csv: str, model_name: str ="SLSelfDriveModel", continue_model=None,
                num_epochs: int = 5, batch_size: int = 1, amp_on: bool = False, dev: str = "cuda:0"):
                
    """
    Run training loop for the SelfDrive model. This function spins up a tensorboard which can be viewed during training. 
    The predictions and ground truth will only be shown during validation. Validation happens every epoch.

    :param train_img_dir The directory containing the training images (Base folder for the "images xx-xx-xx" folders)
    :param train_actions_csv The path to the training csv containing the actions and corresponding image name in format: images 22-03-2022 13-26-11/1647952034.1168804.jpg
    :param valid_img_dir: The directory containing the validation images (Base folder for the "images xx-xx-xx" folders)
    :param valid_actions_csv The path to the validation csv containing the actions and corresponding image name in format: images 22-03-2022 13-26-11/1647952034.1168804.jpg
    :param model_name The name of the model to be trained. This is the name the model will be saved under. Default SLSelfDriveModel
    :param continue_model Path to an existing model that needs to be loaded for continuing training (from assets/models/)
    :param num_epochs The amount of epochs to train for
    :param batch_size The number of images in a batch
    :param amp_on Boolean if AMP should be enabled (should improve training speed, however it is experimental and your mileage may vary) Source: https://pytorch.org/docs/stable/amp.html
    :param dev The device to run the pipeline on, default GPU (cuda:0)
    """
                
    train_loader = get_dataloader(img_folder=train_img_dir, act_csv=train_actions_csv, batch_size=batch_size, normalize=True, normalize_std=False,
                                    motion_blur=False, random_gamma=False, flip=True, rotate=False) # set transforms to true here for data augmentation (only in training!)
    valid_loader = get_dataloader(img_folder=valid_img_dir, act_csv=valid_actions_csv, batch_size=batch_size, normalize=True, normalize_std=False)
    run = True

    model = SelfDriveModel(gpu=True)

    if continue_model:
        model.load_state_dict(torch.load(f"./assets/models/{continue_model}.pt", map_location=dev))
        print("Loaded model!")

    model.to(dev)
    model.train()

    if amp_on:
        scaler = torch.cuda.amp.GradScaler()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1)

    # Make an unique tensorboard name so it wont override your previous one
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logdir_path = os.path.join("src/MachineLearning/SupervisedLearning/tensorboards/", model_name + "_" +
                                now, "tensorboard_training_log")

    # in cmd: tensorboard --logdir="<directory name>" to look back at the tensorboard afterwards
    writer = create_tb(log_dir=logdir_path, wait=True)

    loss_fn = torch.nn.MSELoss()

    print("Training...")
    for epoch in range(0, num_epochs):
        loss_sum, loss_cnt  = 0, 0

        for i, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            input_images, actions = batch['image'].to(dev), batch['actions'].to(dev)
            
            if amp_on:
                with torch.cuda.amp.autocast():
                    outputs = model(input_images)
                    loss = loss_fn(outputs, actions)

                scaler.scale(outputs=loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                scaler.step(optimizer=optimizer)
                scaler.update()
            else:
                outputs = model(input_images)
                loss = loss_fn(outputs, actions)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
            

            loss_sum += loss.cpu().item()
            loss_cnt += 1

        avg_loss = loss_sum / loss_cnt

        tb_show_loss(avg_loss, epoch, "tb_training_loss", writer)

        # Tune learning rate
        scheduler.step(avg_loss)

        if epoch % 1 == 0:
            model.eval()
            run = run_validation(valid_loader=valid_loader, model=model, writer=writer, epoch=epoch, model_name=model_name, now=now, dev=dev)
            model.train()
        
        if not run:
            return f"{model_name}_{now}"

    return f"{model_name}_{now}"


@static_var(best_loss=99999, no_improvement_count=0)
@torch.no_grad()
def run_validation(valid_loader: torch.utils.data.DataLoader, model: torch.nn.Module, epoch: int, writer: SummaryWriter, model_name: str,  now: str, dev: str):
    """
    Validate the model during training. The images, along with predictions and ground truth will be shown in tensorboard.
    Save model if the current loss is lower than the previous lowest loss. Stop training if there was no improvement in validation los for 10 epochs.

    :param valid_loader The validation dataloader
    :param model The model to validate
    :param epoch The current epoch
    :param writer The writer that writes to your current training tensorboard
    :param model_name The name of the model
    :param now Timestamp at start of the pipeline used for saving model
    :param dev The device to run the pipeline on, default GPU (cuda:0)
    """
    loss_fn = torch.nn.MSELoss()
    loss_sum, loss_cnt = 0, 0
    for i, batch in enumerate(tqdm(valid_loader)):

        input_images, actions = batch['image'].to(dev), batch['actions'].to(dev)
        outputs = model(input_images)

        # Show all images and predictions from validation batch
        for idx, image in enumerate(input_images):
            # send image to cpu, make it a numpy array and undo normalization for visualization purposes
            np_image = image.cpu().numpy()
            np_image = ((np_image / np.max(np_image)) * 255).astype(np.uint8)

            # Show image and prediction from the validation batch
            img_with_data = draw_pred_and_target_npy(np_image, filename=batch['img_names'][idx][66:], predicted_actions=outputs[idx], target_actions=actions[idx], dataformats="CHW")
            step = (len(input_images) * i)+idx
            tb_show_image(img_with_data, epoch=step, name=f"Validation images epoch {epoch}", dataformats="HWC", writer=writer) # TODO: Fix the step (epoch) of this, at the moment it does not show all images

        loss = loss_fn(outputs, actions)

        loss_sum += loss.cpu().item()
        loss_cnt += 1

    avg_loss = loss_sum / loss_cnt

    tb_show_loss(avg_loss, epoch, "tb_validation_loss", writer)

    if avg_loss < run_validation.best_loss:      
        model_dir = f"assets/models/{model_name}_{now}.pt"
        torch.save(model.state_dict(), model_dir)
        print(f"\033[92mSaving model {model_name} {now} at epoch {epoch} with loss {avg_loss}\033[0m")
        
        # For examining the model after training has finished, write saved model loss and epoch to tensorboard
        tb_show_text(text=f"Saved model at epoch: {epoch} with loss {avg_loss}", name="saved_model_data", writer=writer)

        run_validation.best_loss = avg_loss
        run_validation.no_improvement_count = 0
    else:
        run_validation.no_improvement_count += 1

    # early stopping afer 10 epochs of no improvement
    if run_validation.no_improvement_count > 9:
        return False
    return True


@torch.no_grad()
def run_testing(test_img_dir: str, test_actions_csv: str, model_name: str ="SLSelfDriveModel", tb_name: str = "tensorboard_testing", wait: bool =True, dev: str ="cpu"):
   
    """
    Test the model. The loss and all of the images in the test set along with predictions and ground truth will be shown in tensorboard.

    :param test_img_dir The directory containing the images (Base folder for the "images xx-xx-xx" folders)
    :param test_actions_csv The path to the testing csv containing the actions and corresponding image name
    :param model_name The name of the model to be tested. If you run testing right after training you can use trained_model_name (return value of run_training), otherwise insert a string with the model name
    :param tb_name The name of the tensorboard. This can be used to distinguish between testing sets.
    :param wait Boolean if the program should be kept running to continue showing the tensorboard after testing is finished
    :param dev The device to run the pipeline on, default CPU
    """
    test_loader = get_dataloader(img_folder=test_img_dir, act_csv=test_actions_csv, batch_size=1, normalize=True, normalize_std=False)

    if dev == "cpu":
        model = SelfDriveModel(gpu=False)
    else:
        model = SelfDriveModel(gpu=True)

    model.load_state_dict(torch.load(f"assets/models/{model_name}.pt", 
                            map_location=dev))
    model.eval()
    model.to(dev)

    loss_sum, loss_cnt = 0, 0

    logdir_path = os.path.join("src/MachineLearning/CANRacing/tensorboards/", model_name, tb_name)
    writer = create_tb(log_dir=logdir_path, wait=wait)

    print("Testing...")
    for idx, batch in enumerate(tqdm(test_loader)):
        img_name = batch['img_names']

        np_image = cv2.imread(img_name[0])
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

        # undo normalization for visualization
        np_image = ((np_image / np.max(np_image)) * 255).astype(np.uint8)
        np_image = np_image[160:325,0:848] # crop to match visualization to what the nn sees

        input_images, actions = batch['image'].to(dev), batch['actions'].to(dev)

        # print("action", actions)
        # print("action", actions.dtype)

        outputs = model(input_images)

        # print("output", outputs)
        # print("output",outputs.dtype)

        # draw image name, prediction and target on image
        img_with_data = draw_pred_and_target_npy(np_image, filename=img_name[0][-49:], predicted_actions=outputs[0], target_actions=actions[0], dataformats="HWC")
        
        # show image, image name, prediction and target in tensorboard
        tb_show_image(img=img_with_data, epoch=idx, name="Test images", dataformats="HWC", writer=writer)

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(outputs, actions)

        loss_sum += loss.cpu().item()
        loss_cnt += 1
        
    avg_loss = loss_sum / loss_cnt
    tb_show_text(text=f"Testing loss (MSE): {avg_loss}", name="tb_testing_loss", writer=writer)

    if wait:
        wait_forever("Press CTRL+C to close tensorboard.")
    return


def run(training=False, test_all=True, debug_training=False, debug_testing=False, trace=False):
    torch.cuda.empty_cache()

    if training:
        trained_model_name = run_training(
                    # 2022
                    train_img_dir="C:/Users/Sabine/Documents/SDC/SL_data/dataset_2022/training/", 
                    train_actions_csv="C:/Users/Sabine/Documents/SDC/SL_data/dataset_2022/training/2022_all_images_smoothed.csv", # smoothed version of the csv

                    # all use the same validation set
                    valid_img_dir="C:/Users/Sabine/Documents/SDC/SL_data/validation/", 
                    valid_actions_csv="C:/Users/Sabine/Documents/SDC/SL_data/validation/40p_data images 30-03-2022 15-17-40_smoothed.csv", # smoothed version of the csv
                    model_name="SLSelfDriveModel", continue_model="", num_epochs=100, amp_on=False, batch_size=16 , dev="cuda:0")
        print("Done training!")
        torch.cuda.empty_cache() # try to free up GPU memory

    if debug_training:
        # ----------------------- DEBUG SETS ----------------------
        trained_model_name = run_training(
                    # small dataset ~200imgs
                    train_img_dir="C:/Users/Sabine/Documents/SDC/SL_data/bigger_test_dataset/training/", 
                    train_actions_csv="C:/Users/Sabine/Documents/SDC/SL_data/bigger_test_dataset/training/training_all.csv",
                    valid_img_dir="C:/Users/Sabine/Documents/SDC/SL_data/bigger_test_dataset/validation/", 
                    valid_actions_csv="C:/Users/Sabine/Documents/SDC/SL_data/bigger_test_dataset/validation/new_validation.csv",

                    # 8 image dataset
                    # train_img_dir="C:/Users/Sabine/Documents/SDC/SL_data/test_dataset",
                    # train_actions_csv="C:/Users/Sabine/Documents/SDC/SL_data/test_dataset/test_csv.csv",
                    # valid_img_dir="C:/Users/Sabine/Documents/SDC/SL_data/test_dataset",
                    # valid_actions_csv="C:/Users/Sabine/Documents/SDC/SL_data/test_dataset/test_csv.csv",
                    model_name="test_steermodel", num_epochs=100, amp_on=False, batch_size=16 , dev="cuda:0")

    if test_all:
        # if you run testing right after training you can use trained_model_name for the model_name parameter, otherwise insert a string with the model name from assets/models
        trained_model_name = ""

        # test set (30-03-2022 15-17-40)
        run_testing(test_img_dir="C:/Users/Sabine/Documents/SDC/SL_data/testing/", 
                    test_actions_csv="C:/Users/Sabine/Documents/SDC/SL_data/testing/testing_60p_data_images_30-03-2022_15-17-40_smoothed.csv", # smoothed version of the csv
                    model_name=trained_model_name, tb_name="tensorboard_testing_test", wait=False, dev="cpu")
        
        # mirrored set (12-04-2022 12-20-39)
        # run_testing(test_img_dir="C:/Users/Sabine/Documents/SDC/SL_data/mirror/", 
        #             test_actions_csv="C:/Users/Sabine/Documents/SDC/SL_data/mirror/final_data images 12-04-2022 12-20-39.csv",
        #             model_name=trained_model_name, tb_name="tensorboard_testing_mirror", wait=False, dev="cuda:0")
        
        # test + mirrored set 
        # run_testing(test_img_dir="C:/Users/Sabine/Documents/SDC/SL_data/testing/", 
        #             test_actions_csv="C:/Users/Sabine/Documents/SDC/SL_data/testing/mirrorlap_final_60p_data images 30-03-2022 15-17-40.csv",
        #             model_name=trained_model_name, tb_name="tensorboard_testing_testandmirror", wait=False, dev="cuda:0")

        # catagorized tests (30-03-2022 15-17-40)
        # straight set
        # run_testing(test_img_dir="C:/Users/Sabine/Documents/SDC/SL_data/Testing_recht/", 
        #             test_actions_csv="C:/Users/Sabine/Documents/SDC/SL_data/Testing_recht/Testing_recht.csv",
        #             model_name=trained_model_name, tb_name="tensorboard_testing_test_straight", wait=False, dev="cuda:0")
        
        # left set
        # run_testing(test_img_dir="C:/Users/Sabine/Documents/SDC/SL_data/Testing_bochten_links/", 
        #             test_actions_csv="C:/Users/Sabine/Documents/SDC/SL_data/Testing_bochten_links/Testing_bochten_links.csv",
        #             model_name=trained_model_name, tb_name="tensorboard_testing_test_left", wait=False, dev="cuda:0")
        
        # right set
        # run_testing(test_img_dir="C:/Users/Sabine/Documents/SDC/SL_data/Testing_bochten_rechts/", 
        #             test_actions_csv="C:/Users/Sabine/Documents/SDC/SL_data/Testing_bochten_rechts/Testing_bochten_rechts.csv",
        #             model_name=trained_model_name, tb_name="tensorboard_testing_test_right", wait=False, dev="cuda:0")
        # print("Done testing!")
    
    if debug_testing:
        # ----------------------- DEBUG SETS ----------------------
        run_testing(
                    # small dataset ~200imgs
                    test_img_dir="C:/Users/Sabine/Documents/SDC/SL_data/bigger_test_dataset/testing/",
                    test_actions_csv="C:/Users/Sabine/Documents/SDC/SL_data/bigger_test_dataset/testing/new_testing.csv",

                    # 8 image dataset
                    # test_img_dir="C:/Users/Sabine/Documents/SDC/SL_data/test_dataset", 
                    # test_actions_csv="C:/Users/Sabine/Documents/SDC/SL_data/test_dataset/test_csv.csv", 
                    model_name="DEBUG", tb_name="tensorboard_testing_debug", wait=True, dev="cpu")
    
    if trace:
        pass
        # TODO: Integrate tracing and saving traced model once training is fully done (Only needed for running model in C++)

    print("Done!")


if __name__ == "__main__":
    run(training=False, test_all=True)
