"""This script fine-tunes ResNet-50 pre-trained on ImageNet to the Stanford Car Dataset"""
import os

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import MultiplicativeLR
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, RandomRotation, ToTensor
from tqdm import tqdm

from feature_vis.transforms import ImagenetNorm
from dataset import CarModels
from train_utils import *

# define the constants
IMAGE_SHAPE = (400, 400)
P_TRAIN = 0.8
BATCH_SIZE = 100
LEARNING_RATE = 0.0005
LR_DECAY = 0.9
NUM_EPOCHS = 30

if __name__ == '__main__':
    # root directory of the car dataset
    root_dir = '/home/mchobanyan/data/stanford-cars/'

    transforms = Compose([
        RandomCrop(IMAGE_SHAPE, pad_if_needed=True),
        RandomHorizontalFlip(),
        RandomRotation(30),
        ToTensor(),
        ImagenetNorm()
    ])

    # define the dataset class
    dataset = CarModels(root_dir, transforms=transforms)
    num_cars = len(dataset)
    num_car_models = len(set(dataset.label_map.values()))

    # randomly split the dataset into train and test
    num_train = int(P_TRAIN * num_cars)
    num_test = num_cars - num_train
    train_data, test_data = random_split(dataset, [num_train, num_test])

    # set up the train and test data loaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # load ResNet-50 with every layer frozen except for layer3-bottleneck5 and beyond,
    # and a new fully-connected network which outputs a 196-dim vector
    device = get_device()
    model = load_resnet50_layer3_bottleneck5(num_car_models)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: LR_DECAY)

    # set up the output logger
    output_dir = '/home/mchobanyan/data/research/transfer/vis/finetune-car-resnet50'
    model_dir = os.path.join(output_dir, 'models')
    create_folder(model_dir)
    logger = TrainingLogger(filepath=os.path.join(output_dir, 'training-log.csv'))

    for epoch in tqdm(range(NUM_EPOCHS)):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
        scheduler.step()
        logger.add_entry(epoch, train_loss, test_loss, train_acc, test_acc)
        checkpoint(model, os.path.join(model_dir, f'model_epoch{epoch}.pt'))
