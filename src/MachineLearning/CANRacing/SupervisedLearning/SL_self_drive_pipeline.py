# To update requirements.txt: https://github.com/bndr/pipreqs

# SETUP THE PIPELINE:
# Download python 3.7.13
# Create a conda environment and run: conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch 
# to prevent "Torch not compiled with CUDA enabled" (This step is only needed if you plan on training on GPU)
# Then run pip install requirements.txt

# TODO: 
# 1. Rescale input images to smaller size✔️, then crop to remove unnecessary data
# 2. Sort out images for training set (variation in images, several datasets)
# 3. Optimize pipeline speed https://pytorch.org/docs/stable/amp.html (✔️ implemented but not sure if it works)
# 4. Remove brake from the NN (will have to remove brake from .csv for this)
# 5. Limit NN output values https://discuss.pytorch.org/t/how-to-return-output-values-only-from-0-to-1/24517/5
# 6. Reduce size of the NN? (layers) ✔️
# 7. Spin up a tensorboard with metrics and image + prediction (for reviewing training/testing) ✔️

import torch
from load_data import get_dataloader
from tqdm import tqdm
from SelfDriveModel import SelfDriveModel
from utilities import static_var, wait_forever
import numpy as np
from tensorboardfuncs import create_tb, tb_show_text, tb_show_loss, tb_show_image
from tbprep import draw_pred_and_traget_npy
import skimage.io 

def run_training(train_img_dir: str, train_actions_csv: str, valid_img_dir: str, valid_actions_csv: str, model_name="SLSelfDriveModel",
                num_epochs: int = 5, batch_size: int = 1, amp_on = False, dev: str = "cuda:0"):
                
    train_loader = get_dataloader(img_folder=train_img_dir, act_csv=train_actions_csv, batch_size=batch_size, normalize=True) # set transforms to true here for data augmentation (only in training!)
    valid_loader = get_dataloader(img_folder=valid_img_dir, act_csv=valid_actions_csv, batch_size=batch_size, normalize=True)

    model = SelfDriveModel()
    model.to(dev)
    model.train()

    if amp_on:
        scaler = torch.cuda.amp.GradScaler()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1)

    # in cmd: tensorboard --logdir="<directory name>" to look back at the tensorboard
    writer = create_tb(tb_name="tb_training_log", log_dir="src/MachineLearning/CANRacing/tensorboard_training_log", wait=True)

    loss_fn = torch.nn.MSELoss()

    print("Training...")
    for epoch in range(0, num_epochs):
        loss_sum, loss_cnt  = 0, 0

        for i, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            input_images, actions = batch['image'].to(dev), batch['actions'].to(dev)
            
            if amp_on:
                # AMP IS EXPERIMENTAL FOR FASTER TRAINING
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
            

            loss_sum += loss.item()
            loss_cnt += 1

        avg_loss = loss_sum / loss_cnt

        tb_show_loss(avg_loss, epoch, "training_loss", writer)

        # Tune learning rate
        scheduler.step(avg_loss, epoch)

        if epoch % 2 == 0:
            model.eval()
            run_validation(valid_loader=valid_loader, model=model, writer=writer, epoch=epoch, dev=dev, model_name=model_name)
            model.train()
    return


@static_var(best_loss=99999)
@torch.no_grad()
def run_validation(valid_loader, model, writer, epoch, dev, model_name):
    loss_fn = torch.nn.MSELoss()
    loss_sum, loss_cnt = 0, 0
    for i, batch in enumerate(tqdm(valid_loader)):

        input_images, actions = batch['image'].to(dev), batch['actions'].to(dev)

        outputs = model(input_images)

        if loss_cnt == 0:
            # TODO: show validation images in TB
            # tb_show_image(img=img_with_data, epoch=idx, name="Test images", dataformats="HWC", writer=writer)
            pass

        loss = loss_fn(outputs, actions)

        loss_sum += loss.cpu().item()
        loss_cnt += 1

    avg_loss = loss_sum / loss_cnt

    tb_show_loss(avg_loss, epoch, "tb_validation", "loss", writer)

    if avg_loss < run_validation.best_loss:      
        torch.save(model.state_dict(), f"src/MachineLearning/CANRacing/models/{model_name}.pt")
        print(f"\033[92mSaving model at epoch {epoch} with loss {avg_loss}\033[0m")

        run_validation.best_loss = avg_loss
    return


@torch.no_grad()
def run_testing(test_img_dir: str, test_actions_csv: str, model_name="SLSelfDriveModel", wait=True, dev="cpu"):
    test_loader = get_dataloader(img_folder=test_img_dir, act_csv=test_actions_csv, batch_size=1, normalize=True)

    # maybe redundant
    device = torch.device("cpu")

    model = SelfDriveModel()
    model.load_state_dict(torch.load(f"src/MachineLearning/CANRacing/models/{model_name}.pt", 
                            map_location=device))
    model.eval()
    model.to(dev)

    loss_sum, loss_cnt = 0, 0
    writer = create_tb(log_dir="src/MachineLearning/CANRacing/tensorboard_testing_log", wait=wait)

    print("Testing...")
    for idx, batch in enumerate(tqdm(test_loader)):
        img_name = batch['img_names']

        np_image = skimage.io.imread(img_name[0])
        np_image = ((np_image / np.max(np_image)) * 255).astype(np.uint8)

        input_images, actions = batch['image'].to(device), batch['actions'].to(device)

        outputs = model(input_images)

        # draw image name, prediction and target on image
        img_with_data = draw_pred_and_traget_npy(np_image, filename=img_name[0][66:], predicted_actions=outputs, target_actions=actions, dataformats="HWC")
        
        # show image, image name, prediction and target in tensorboard
        tb_show_image(img=img_with_data, epoch=idx, name="Test images", dataformats="HWC", writer=writer)

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(outputs, actions)

        loss_sum += loss.item()
        loss_cnt += 1
        
    avg_loss = loss_sum / loss_cnt
    tb_show_text(avg_loss, idx, name="tb_testing", writer=writer)

    if wait:
        wait_forever("Press CTRL+C to close tensorboard.")
    return


def run(training=False, experiment=False):
    if training:
        run_training(
                    train_img_dir="C:/Users/Sabin/Documents/SDC/SL_data/dataset_only_turns/training", 
                    train_actions_csv="C:/Users/Sabin/Documents/SDC/SL_data/dataset_only_turns/training/train_data_images_18-11-2021_14-59-21_2.csv",
                    
                    # 8 IMAGE DATASET FOR DEBUGGING
                    # train_img_dir="C:/Users/Sabin/Documents/SDC/SL_data/test_set",
                    # train_actions_csv="C:/Users/Sabin/Documents/SDC/SL_data/test_set/test_csv.csv",

                    valid_img_dir="C:/Users/Sabin/Documents/SDC/SL_data/dataset_only_turns/validation", 
                    valid_actions_csv="C:/Users/Sabin/Documents/SDC/SL_data/dataset_only_turns/validation/val_data_images_18-11-2021_15-12-21_2.csv",
                    
                    # 8 IMAGE DATASET FOR DEBUGGING
                    # valid_img_dir="C:/Users/Sabin/Documents/SDC/SL_data/test_set",
                    # valid_actions_csv="C:/Users/Sabin/Documents/SDC/SL_data/test_set/test_csv.csv",
                    model_name="SLSelfDriveModel1", num_epochs=100, amp_on=False, batch_size=4, dev="cuda:0")

        # try to free up GPU memory
        # torch.cuda.empty_cache()

    run_testing(test_img_dir="C:/Users/Sabin/Documents/SDC/SL_data/dataset_only_turns/validation", 
                test_actions_csv="C:/Users/Sabin/Documents/SDC/SL_data/dataset_only_turns/validation/val_data_images_18-11-2021_15-12-21_2.csv",
                model_name="SLSelfDriveModelBochten140522", wait=True, dev="cpu") # test on cpu

    # 8 IMAGE DATASET FOR DEBUGGING
    # run_testing(test_img_dir="C:/Users/Sabin/Documents/SDC/SL_data/test_set", 
    #             test_actions_csv="C:/Users/Sabin/Documents/SDC/SL_data/test_set/test_csv.csv", 
    #             model_name="SLSelfDriveModel8IMG", dev="cpu") 

    if experiment:
        # 90% turns
        run_training(
                train_img_dir="C:/Users/Sabin/Documents/SDC/SL_data/dataset_90p_turns/training", 
                train_actions_csv="C:/Users/Sabin/Documents/SDC/SL_data/dataset_90p_turns/training/train_data_images_18-11-2021_14-59-21_2.csv",
                valid_img_dir="C:/Users/Sabin/Documents/SDC/SL_data/dataset_90p_turns/validation", 
                valid_actions_csv="C:/Users/Sabin/Documents/SDC/SL_data/dataset_90p_turns/validation/val_data_images_18-11-2021_15-12-21_2.csv",
                model_name="SLSelfDriveModel90p", num_epochs=35, amp_on=False, batch_size=4, dev="cuda:0")

        torch.cuda.empty_cache()

        # 95% turns
        run_training(
                train_img_dir="C:/Users/Sabin/Documents/SDC/SL_data/dataset_90p_turns/training", 
                train_actions_csv="C:/Users/Sabin/Documents/SDC/SL_data/dataset_95p_turns/training/train_data_images_18-11-2021_14-59-21_2.csv",
                valid_img_dir="C:/Users/Sabin/Documents/SDC/SL_data/dataset_90p_turns/validation", 
                valid_actions_csv="C:/Users/Sabin/Documents/SDC/SL_data/dataset_95p_turns/validation/val_data_images_18-11-2021_15-12-21_2.csv",
                model_name="SelfDriveModel95p", num_epochs=35, amp_on=False, batch_size=4, dev="cuda:0")

    print("Done!")


if __name__ == "__main__":
    run(training=False, experiment=False)
