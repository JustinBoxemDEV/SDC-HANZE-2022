import torch
from load_data import get_dataloader
import tqdm
import numpy as np
from SelfDriveModel import SelfDriveModel

@torch.no_grad()
def run_training(train_img_dir: str, train_actions_csv: str, valid_img_dir: str, valid_actions_csv: str, 
                num_epochs: int = 5, batch_size: int = 8, dev: str = "cuda:0"):
                
    train_loader = get_dataloader(img_folder=train_img_dir, act_csv=train_actions_csv, batch_size=batch_size)
    # valid_loader = get_dataloader(img_folder=valid_img_dir, act_csv=valid_actions_csv, batch_size=batch_size)

    # model = None
    model = SelfDriveModel()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # scheduler = optimizer

    last_loss = 0
    total_loss = 0

    print("Training...")
    for epoch in range(0, num_epochs):
        # print(f"Executing epoch {epoch}!")

        for idx, batch in enumerate(train_loader):
            # print(f"Executing batch number {idx}!")

            # print("Images in batch:", batch['image'])
            # print("Ground truth actions in batch:", batch['actions'])

            input_images, actions = batch['image'].to(dev), batch['actions'].to(dev)

            optimizer.zero_grad()

            # TODO: use moddel outputs instead of test data
            # outputs = model(input_images)
            
            # example output for testing pipeline
            outputs = [[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],
                        [0., 0., 0.],[0., 0., 0.],[0., 0., 0.],
                        [0., 0., 0.],[0., 0., 0.]]
            outputs = np.array(outputs)
            outputs = torch.from_numpy(outputs)
            outputs = outputs.to(dev)

            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs, actions)
            # loss.backward()

            optimizer.step()

            total_loss += loss.item()

            if idx % 1000 == 999:
                last_loss = total_loss / 1000 # loss per batch
                print(f'Batch {idx+1} loss: {last_loss}')
                total_loss = 0
        
        # if epoch % 2 == 0:
        #     # run_validation(valid_loader=valid_loader)
        #     pass
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
        
        # TODO: model prediction
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
                    num_epochs=1, batch_size=8, dev="cuda:0")

    # try to free up GPU memory
    torch.cuda.empty_cache()
    run_testing(test_img_dir="C:/Users/Sabin/Documents/SDC/SL_data/testing/", 
                test_actions_csv="C:/Users/Sabin/Documents/SDC/SL_data/testing/douwe_data_images_18-11-2021 15-12-21.csv")

    print("Done!")


if __name__ == "__main__":
    run(training=True)