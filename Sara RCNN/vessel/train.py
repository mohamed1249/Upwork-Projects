# Based on Pytorch tutorial for transfer learning
# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import argparse
import logging

cudnn.benchmark = True

# Data augmentation and normalization for training
# Just normalization for validation
def parse_args():
    parser = argparse.ArgumentParser(description='Train Binary Classifier')
    parser.add_argument('--data_dir', type=str, default='images', help='data directory')
    parser.add_argument('--model_dir', type=str, default='models/',help='model directory')
    parser.add_argument('--num_epochs', default=20, type=int, help='number of epochs for training')
    parser.add_argument('--batch_size', default=12, type=int, help='batch size for training')
    parser.add_argument('--step_size', default=7, type=int, help='number of steps after which to reduce learning rate')
    parser.add_argument('--gamma', default=0.2, type=float, help='factor to multiply learning rate')
    parser.add_argument('--lr', default=0.0003, type=float, help='learning rate')
    args = parser.parse_args()
        # assert not os.path.exists(cfg.train.model_dir), "{} already exists".format(cfg.train.model_dir)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    return args

def getlogger(mode_dir):
    logger = logging.getLogger('Binary_classifier')
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(os.path.join(mode_dir, 'log.txt' ))
    handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

def train_model(model: object,
                criterion: object,
                optimizer: object,
                scheduler: object,
                num_epochs: int,
                logger: object,
                model_dir: object):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    print(f"Using device: {device}")
    logger.info(f"Using device: {device}")
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch}/{num_epochs - 1}')
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            logger.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                model_path = os.path.join(model_dir, f"best_acc.pth")
                logger.info(f"Saving model {model_path}")
                torch.save(model.state_dict(), model_path)


        print()

    time_elapsed = time.time() - since
    logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best val Acc: {best_acc:4f}')
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')




if __name__ == '__main__':
    args = parse_args()

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomApply(torch.nn.ModuleList([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                transforms.RandomAffine(5)]), p=0.5),
            transforms.RandomVerticalFlip(),
            transforms.RandomInvert(0.2),
            transforms.RandomSolarize(threshold=0.8, p=0.2),
            transforms.RandomPosterize(bits=2),
            transforms.GaussianBlur(kernel_size=5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                 shuffle=True, num_workers=2)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Use pretrained model
    model_ft = models.resnet34(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.AdamW(model_ft.parameters(), lr=args.lr)

    # Decay LR by a factor of 0.2 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,
                                           step_size=args.step_size,
                                           gamma=args.gamma)
    logger = getlogger(args.model_dir)

    # Train model
    train_model(model=model_ft,
                criterion=criterion,
                optimizer=optimizer_ft,
                scheduler=exp_lr_scheduler,
                num_epochs=args.num_epochs,
                logger=logger,
                model_dir=args.model_dir)

