# To update requirements.txt: https://github.com/bndr/pipreqs

# Used with python 3.7.13
# First run: conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch 
# to prevent "Torch not compiled with CUDA enabled" 
# Then run pip install requirements.txt

# TODO: 
# 1. rescale input images to smaller size
# 2. sort out images for training set (variation in images)
# 3. Optimize pipeline (speed) https://pytorch.org/docs/stable/amp.html

import torch
from load_data import get_dataloader
from tqdm import tqdm
from SelfDriveModel import SelfDriveModel
from utilities import static_var


def run_training(train_img_dir: str, train_actions_csv: str, valid_img_dir: str, valid_actions_csv: str, 
                num_epochs: int = 5, batch_size: int = 1, amp_on = False, dev: str = "cuda:0"):
                
    train_loader = get_dataloader(img_folder=train_img_dir, act_csv=train_actions_csv, batch_size=batch_size, normalize=True) # set transforms to true here for data augmentation
    valid_loader = get_dataloader(img_folder=valid_img_dir, act_csv=valid_actions_csv, batch_size=batch_size, normalize=True)

    model = SelfDriveModel()
    model.to(dev)
    model.train()

    if amp_on:
        scaler = torch.cuda.amp.GradScaler()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.1)

    loss_fn = torch.nn.MSELoss()

    print("Training...")
    for epoch in range(0, num_epochs):
        loss_sum, loss_cnt  = 0, 0

        for idx, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            input_images, actions = batch['image'].to(dev), batch['actions'].to(dev)
            
            if amp_on:
                # AMP IS EXPERIMENTAL FOR FASTER TRAINING, SET IT TO FALSE DURING TRAINING
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

        # Tune learning rate
        scheduler.step(avg_loss, epoch)

        print(f"Avg loss on epoch {epoch} is {avg_loss}")

        if epoch % 1 == 0:
            model.eval()
            run_validation(valid_loader=valid_loader, epoch=epoch, model=model, dev=dev)
            model.train()
    return


@static_var(best_loss=99999)
@torch.no_grad()
def run_validation(valid_loader, model, epoch, dev):
    loss_sum, loss_cnt = 0, 0
    for _, batch in enumerate(tqdm(valid_loader)):

        input_images, actions = batch['image'].to(dev), batch['actions'].to(dev)

        outputs = model(input_images)

        # TODO: show image in tensorboard

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(outputs, actions)

        loss_sum += loss.cpu().item()
        loss_cnt += 1

    avg_loss = loss_sum / loss_cnt

    if avg_loss < run_validation.best_loss:      
        torch.save(model.state_dict(), "src/MachineLearning/CANRacing/models/SLSelfDriveModel.pt")
        
        print(f"Saving model at epoch {epoch} with loss {avg_loss}")
        run_validation.best_loss = avg_loss
    return


@torch.no_grad()
def run_testing(test_img_dir: str, test_actions_csv: str, dev="cuda:0"):
    test_loader = get_dataloader(img_folder=test_img_dir, act_csv=test_actions_csv, batch_size=1, normalize=True)

    device = torch.device("cpu")
    model = SelfDriveModel()
    model.load_state_dict(torch.load("src/MachineLearning/CANRacing/models/SLSelfDriveModel.pt", 
                            map_location=device))
    
    model.eval()
    model.to(dev)

    loss_sum, loss_cnt = 0, 0

    print("Testing...")
    for idx, batch in enumerate(tqdm(test_loader)):
        input_images, actions = batch['image'].to(device), batch['actions'].to(device)

        outputs = model(input_images)

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(outputs, actions)

        # print(f"Batch loss: {loss}")
        loss_sum += loss.cpu().item()
        loss_cnt += 1
        
    avg_loss = loss_sum / loss_cnt
    print("Avg loss:", avg_loss)

    return


def run(training=False):
    if training:
        run_training(
                    train_img_dir="C:/Users/Sabin/Documents/SDC/SL_data/training/", 
                    train_actions_csv="C:/Users/Sabin/Documents/SDC/SL_data/training/douwe_data_images_18-11-2021_14-59-21_2.csv",
                    
                    # 8 IMAGE DATASET FOR DEBUGGING
                    # train_img_dir="C:/Users/Sabin/Documents/SDC/SL_data/test_set",
                    # train_actions_csv="C:/Users/Sabin/Documents/SDC/SL_data/test_set/test_csv.csv",

                    valid_img_dir="C:/Users/Sabin/Documents/SDC/SL_data/validation/", 
                    valid_actions_csv="C:/Users/Sabin/Documents/SDC/SL_data/validation/douwe_data_images_18-11-2021 15-12-21.csv",
                    
                    # 8 IMAGE DATASET FOR DEBUGGING
                    # valid_img_dir="C:/Users/Sabin/Documents/SDC/SL_data/test_set",
                    # valid_actions_csv="C:/Users/Sabin/Documents/SDC/SL_data/test_set/test_csv.csv",
                    num_epochs=100, amp_on=False, batch_size=2, dev="cuda:0")

        # try to free up GPU memory
        torch.cuda.empty_cache()

    # run_testing(test_img_dir="C:/Users/Sabin/Documents/SDC/SL_data/testing/", 
    #             test_actions_csv="C:/Users/Sabin/Documents/SDC/SL_data/testing/data images 18-11-2021 13-45-33", 
    #             dev="cuda:0")

    # 8 IMAGE DATASET FOR DEBUGGING
    # run_testing(test_img_dir="C:/Users/Sabin/Documents/SDC/SL_data/test_set", 
    #             test_actions_csv="C:/Users/Sabin/Documents/SDC/SL_data/test_set/test_csv.csv", 
    #             dev="cpu")

    print("Done!")


if __name__ == "__main__":
    run(training=True)
