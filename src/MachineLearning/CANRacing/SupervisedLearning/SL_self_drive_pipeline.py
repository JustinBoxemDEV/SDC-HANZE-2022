import torch
import skimage.io
from matplotlib import pyplot as plt
from load_data import get_dataloader
import tqdm
import numpy as np
from SelfDriveModel import SelfDriveModel
from utilities import static_var

@torch.no_grad()
def run_training(train_img_dir: str, train_actions_csv: str, valid_img_dir: str, valid_actions_csv: str, 
                num_epochs: int = 5, batch_size: int = 8, dev: str = "cuda:0"):
                
    train_loader = get_dataloader(img_folder=train_img_dir, act_csv=train_actions_csv, batch_size=batch_size, normalize=True) # set transforms to true here for data augmentation
    valid_loader = get_dataloader(img_folder=valid_img_dir, act_csv=valid_actions_csv, batch_size=batch_size, normalize=True)

    model = SelfDriveModel()
    model.to(dev)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # TODO: maybe scheduler = optimizer

    total_loss = 0

    print("Training...")
    for epoch in range(0, num_epochs):
        print(f"Executing epoch {epoch}!")

        for idx, batch in enumerate(train_loader):
            print(f"Executing batch number {idx}!")

            # print("Images in batch:", batch['image'])
            # print("Ground truth actions in batch:", batch['actions'])

            input_images, actions = batch['image'].to(dev), batch['actions'].to(dev)

            optimizer.zero_grad()

            outputs = model(input_images)
            # print("Prediction:", outputs)
            # print("Ground truth:", actions)

            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(outputs, actions)
            # TODO: maybe backward? (loss.backward())

            optimizer.step()

            total_loss += loss.item()

        # if epoch % 2 == 0:
        if True:
            model.eval()
            run_validation(valid_loader=valid_loader, epoch=epoch, model=model, dev=dev)
            model.train()
    return


@static_var(best_loss=99999)
@torch.no_grad()
def run_validation(valid_loader, model, epoch, dev):
    loss_sum, loss_cnt = 0, 0
    for _, batch in enumerate(valid_loader):

        input_images, actions = batch['image'].to(dev), batch['actions'].to(dev)

        outputs = model(input_images)

        # TODO: show image in tensorboard

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(outputs, actions)

        loss_sum += loss.cpu().item()
        loss_cnt += 1

    avg_loss = loss_sum / loss_cnt

    if avg_loss < run_validation.best_loss:
        # save_dict = {"model": model.state_dict(), "loss": avg_loss, "epoch": epoch, "type": str(type(model))}
        # torch.save(save_dict, "C:/Users/Sabin/Documents/vsc_cpp_projects/SDC-stuff/SDC-HANZE-2022/src/MachineLearning/CANRacing/models/SelfDriveModel.pt") # Doesn't work for some reason
        torch.save(model, "C:/Users/Sabin/Documents/vsc_cpp_projects/SDC-stuff/SDC-HANZE-2022/src/MachineLearning/CANRacing/models/SLSelfDriveModel.pt")
        print(f"Saving model with loss: {avg_loss}")
        run_validation.best_loss = avg_loss
    return


@torch.no_grad()
def run_testing(test_img_dir: str, test_actions_csv: str, dev="cuda:0"):
    test_loader = get_dataloader(img_folder=test_img_dir, act_csv=test_actions_csv, batch_size=8, normalize=True)

    # model = SelfDriveModel()
    # model.load_state_dict(torch.load("C:/Users/Sabin/Documents/vsc_cpp_projects/SDC-stuff/SDC-HANZE-2022/src/MachineLearning/CANRacing/models/SLSelfDriveModel.pt"))
    # print(f"Loaded model with loss: {model['loss']}")

    model = torch.load("C:/Users/Sabin/Documents/vsc_cpp_projects/SDC-stuff/SDC-HANZE-2022/src/MachineLearning/CANRacing/models/SLSelfDriveModel.pt")
    model.eval()
    model.to(dev)

    loss_sum, loss_cnt = 0, 0

    print("Testing...")
    for idx, batch in enumerate(test_loader):
        print(f"Executing batch number {idx}!")

        input_images, actions = batch['image'].to(dev), batch['actions'].to(dev)

        outputs = model(input_images)

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(outputs, actions)

        loss_sum += loss.cpu().item()
        loss_cnt += 1
        
    avg_loss = loss_sum / loss_cnt
    print("Avg loss:", avg_loss)

    return


def run(training=False):
    if training:
        run_training(train_img_dir="C:/Users/Sabin/Documents/SDC/SL_data/training/", 
                    train_actions_csv="C:/Users/Sabin/Documents/SDC/SL_data/training/douwe_data_images_18-11-2021_14-59-21_2.csv",
                    # TEST AND VALIDATION DATA IS CURRENTLY THE SAME DUE TO LACK OF DATA
                    valid_img_dir="C:/Users/Sabin/Documents/SDC/SL_data/validation/", 
                    valid_actions_csv="C:/Users/Sabin/Documents/SDC/SL_data/validation/douwe_data_images_18-11-2021 15-12-21.csv",
                    num_epochs=1, batch_size=8, dev="cuda:0")

        # try to free up GPU memory
        # torch.cuda.empty_cache()

    run_testing(test_img_dir="C:/Users/Sabin/Documents/SDC/SL_data/testing/", 
                test_actions_csv="C:/Users/Sabin/Documents/SDC/SL_data/testing/douwe_data_images_18-11-2021 15-12-21.csv", 
                dev="cuda:0")

    print("Done!")


if __name__ == "__main__":
    run(training=False)
