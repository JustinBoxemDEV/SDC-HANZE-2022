"""
Train a YOLOv5 classifier model on a classification dataset
Usage - train:
    cd src/MachineLearning/CANRacing/SupervisedLearning/classification
    python my_classifier.py --epochs 5 --img 128 --batch-size 16 --workers 4
Usage - inference:
    model = torch.load('path/to/best.pt', map_location=torch.device('cpu'))['model'].float()

    Or just use visualize_classification.py :)

Sources:
https://github.com/ultralytics/yolov5/blob/classifier/classifier.py
"""

import argparse
import math
from pathlib import Path
from DirectionClassificationModel import DirectionClassificationModel
from torch.utils.data import random_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as T
from torch.cuda import amp
from tqdm import tqdm
torch.manual_seed(8)

from yolov5.utils.general import (NUM_THREADS, increment_path)
from yolov5.utils.torch_utils import select_device

# Normalization using standard deviation
normalize = lambda x, mean=0.5, std=0.25: (x - mean) / std
denormalize = lambda x, mean=0.5, std=0.25: x * std + mean


def train():
    save_dir, bs, epochs, nw, imgsz = \
        Path(opt.save_dir), opt.batch_size, opt.epochs, min(NUM_THREADS, opt.workers), opt.img_size

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = wdir / 'last.pt', wdir / 'best.pt'
    print("Working directory:", wdir)

    # Transforms
    trainform = T.Compose([
        T.ToTensor(),
        ])  
    testform = T.Compose(trainform.transforms[-2:])

    train_img_dir="D:/KWALI/classification_kwali/"
    # test_img_dir="C:/Users/Sabin/Documents/SDC/SL_data/dataset_2022_3_classes/validation/"

    # Dataloader (random split)
    full_dataset = torchvision.datasets.ImageFolder(root=train_img_dir, transform=trainform)
    dataset_size = len(full_dataset)
    test_size = int(0.1 * dataset_size)
    train_size = dataset_size - test_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # trainset = torchvision.datasets.ImageFolder(root=train_img_dir, transform=trainform)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=nw)
    # testset = torchvision.datasets.ImageFolder(root=test_img_dir, transform=testform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=nw)

    names = ["left","straight","right"] # train_dataset.classes
    nc = len(names)

    print(f'Training model on TTAssen dataset with {nc} classes...')

    # Show images
    images, labels = iter(trainloader).next()
    imshow(denormalize(images[:64]), labels[:64], names=names, f=save_dir / 'train_images.jpg')

    # Model
    model = DirectionClassificationModel(gpu=True)
    # model.load_state_dict(torch.load(f"./assets/models/classification/classification_model.pt", map_location=device)) # load model

    # Optimizer
    lr0 = 0.00001 * bs  # intial lr
    lrf = 0.01  # final lr (fraction of lr0)
    if opt.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr0 / 10)
    elif opt.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr0 / 10)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=0.9, nesterov=True)

    # Scheduler
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Train
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()  # loss function
    best_fitness = 0.0
    # scaler = amp.GradScaler(enabled=cuda)
    print(f'Image sizes {imgsz} train, {imgsz} test\n'
          f'Using {nw} dataloader workers\n'
          f"Logging results to {save_dir}\n"
          f'Starting training for {epochs} epochs...\n\n'
          f"{'epoch':10s}{'gpu_mem':10s}{'train_loss':12s}{'val_loss':12s}{'accuracy':12s}")
    for epoch in range(epochs):  # loop over the dataset multiple times
        mloss = 0.0  # mean loss
        model.train()
        pbar = tqdm(enumerate(trainloader), total=len(trainloader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, (images, labels) in pbar:  # progress bar
            images, labels = resize(images.to(device)), labels.to(device)

            # Forward
            with amp.autocast(enabled=False):  # stability issues when enabled
                loss = criterion(model(images), labels)

            # Backward
            loss.backward()  # scaler.scale(loss).backward()

            # Optimize
            optimizer.step()  # scaler.step(optimizer); scaler.update()
            optimizer.zero_grad()

            # Print
            mloss += loss.item()
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            pbar.desc = f"{'%s/%s' % (epoch + 1, epochs):10s}{mem:10s}{mloss / (i + 1):<12.3g}"

            # Test
            if i == len(pbar) - 1:
                torch.cuda.empty_cache()
                fitness = test(model, testloader, names, criterion, pbar=pbar)  # test

        # Scheduler
        scheduler.step()

        # Best fitness
        if fitness > best_fitness:
            best_fitness = fitness

        # Save model
        final_epoch = epoch + 1 == epochs
        if (not opt.nosave) or final_epoch:
            torch.save(model.state_dict(), last)
            if best_fitness == fitness:
                torch.save(model.state_dict(), best)

    # Train complete
    if final_epoch:
        print(f'Training complete. Results saved to {save_dir}.')

        # Show predictions
        images, labels = iter(testloader).next()
        images = resize(images.to(device))
        pred = torch.max(model(images), 1)[1]
        imshow(denormalize(images), labels, pred, names, verbose=True, f=save_dir / 'test_images.jpg')


def test(model, dataloader, names, criterion=None, verbose=False, pbar=None):
    model.eval()
    pred, targets, loss = [], [], 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = resize(images.to(device)), labels.to(device)
            y = model(images)
            pred.append(torch.max(y, 1)[1])
            targets.append(labels)
            if criterion:
                loss += criterion(y, labels)

    pred, targets = torch.cat(pred), torch.cat(targets)
    correct = (targets == pred).float()

    if pbar:
        pbar.desc += f"{loss / len(dataloader):<12.3g}{correct.mean().item():<12.3g}"

    accuracy = correct.mean().item()
    if verbose:  # all classes
        print(f"{'class':10s}{'number':10s}{'accuracy':10s}")
        print(f"{'all':10s}{correct.shape[0]:10s}{accuracy:10.5g}")
        for i, c in enumerate(names):
            t = correct[targets == i]
            print(f"{c:10s}{t.shape[0]:10s}{t.mean().item():10.5g}")

    return accuracy


def imshow(img, labels=None, pred=None, names=None, nmax=64, verbose=False, f=Path('images.jpg')):
    # Show classification image grid with labels (optional) and predictions (optional)
    import matplotlib.pyplot as plt

    names = names or [f'class{i}' for i in range(1000)]
    blocks = torch.chunk(img.cpu(), len(img), dim=0)  # select batch index 0, block by channels
    n = min(len(blocks), nmax)  # number of plots
    m = min(8, round(n ** 0.5))  # 8 x 8 default
    fig, ax = plt.subplots(math.ceil(n / m), m, tight_layout=True)  # 8 rows x n/8 cols
    ax = ax.ravel() if m > 1 else [ax]
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for i in range(n):
        ax[i].imshow(blocks[i].squeeze().permute((1, 2, 0)))  # cmap='gray'
        ax[i].axis('off')
        if labels is not None:
            s = names[labels[i]] + (f'—{names[pred[i]]}' if pred is not None else '')
            ax[i].set_title(s)

    plt.savefig(f, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"examples saved to {f}")

    if verbose and labels is not None:
        print('True:     ', ' '.join(f'{names[i]:3s}' for i in labels))
    if verbose and pred is not None:
        print('Predicted:', ' '.join(f'{names[i]:3s}' for i in pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128, help='total batch size for all GPUs')
    parser.add_argument('--img-size', type=int, default=128, help='train, test image sizes (pixels)')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='Adam', help='optimizer')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default='./runs/', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    # Parameters
    device = select_device(opt.device, batch_size=opt.batch_size)
    cuda = device.type != 'cpu'
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run
    resize = torch.nn.Upsample(size=(opt.img_size, opt.img_size), mode='bilinear', align_corners=False)  # image resize

    # Train
    train()